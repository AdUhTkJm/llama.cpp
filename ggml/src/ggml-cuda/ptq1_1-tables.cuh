// Host-side cache of the PTQ1_1 pattern tables a kernel needs.
//
// The codebook is model-wide and the escape list is per tensor. Neither fits in the
// tensor buffer, so the loader registers them in ggml-base keyed by tensor and the
// backend keeps a device copy here. One upload per (tensor, device), like get_dense_w().
#pragma once

#include "common.cuh"
#include "ggml-quants.h"

#include <algorithm>
#include <map>
#include <mutex>
#include <tuple>
#include <vector>

// Two device copies of the 767 x 5 byte codebook: the packed bytes and one aligned 64-bit
// word per pattern, which the MMQ tile loader reads with a single load.
static void ptq1_1_device_codebook(const uint8_t * host, int device, const uint8_t ** out, const uint64_t ** out64) {
    static std::mutex mutex;
    static std::map<std::pair<int, const uint8_t *>, std::pair<const uint8_t *, const uint64_t *>> cache;

    std::lock_guard<std::mutex> lock(mutex);
    const auto key = std::make_pair(device, host);
    const auto it  = cache.find(key);
    if (it != cache.end()) {
        *out   = it->second.first;
        *out64 = it->second.second;
        return;
    }

    constexpr size_t n = 767*5;
    uint8_t * dev = nullptr;
    CUDA_CHECK(cudaMalloc(&dev, n));
    CUDA_CHECK(cudaMemcpy(dev, host, n, cudaMemcpyHostToDevice));

    std::vector<uint64_t> wide(767, 0);
    for (size_t i = 0; i < 767; ++i) {
        for (int b = 0; b < 5; ++b) {
            wide[i] |= (uint64_t) host[5*i + b] << (8*b);
        }
    }
    uint64_t * dev64 = nullptr;
    CUDA_CHECK(cudaMalloc((void **) &dev64, wide.size()*sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(dev64, wide.data(), wide.size()*sizeof(uint64_t), cudaMemcpyHostToDevice));

    cache.emplace(key, std::make_pair((const uint8_t *) dev, (const uint64_t *) dev64));
    *out   = dev;
    *out64 = dev64;
}

// w->data, the codebook pointer and the escape count form the key: a reloaded model can
// reuse an address, and a different escape list must not hit the old device buffers.
static ptq1_1_tables ptq1_1_tables_get(const ggml_tensor * w, int device) {
    const uint8_t  * cb_host = nullptr;
    const uint64_t * esc     = nullptr;
    int64_t          n_esc   = 0;
    if (!ggml_ptq1_1_get_escape(w, &cb_host, &esc, &n_esc)) {
        GGML_ABORT("ptq1_1: no pattern tables registered for '%s'", w->name);
    }
    if (cb_host == nullptr) {
        GGML_ABORT("ptq1_1: '%s' has no codebook", w->name);
    }

    using key_t = std::tuple<int, const void *, const uint8_t *, int64_t>;
    static std::mutex mutex;
    static std::map<key_t, ptq1_1_tables> cache;
    const key_t key = std::make_tuple(device, (const void *) w->data, cb_host, n_esc);

    {
        std::lock_guard<std::mutex> lock(mutex);
        const auto it = cache.find(key);
        if (it != cache.end()) {
            return it->second;
        }
    }

    ptq1_1_tables t = {};
    ptq1_1_device_codebook(cb_host, device, &t.codebook, &t.codebook64);
    t.n_esc = (int) n_esc;

    std::vector<uint8_t>  patt(5*n_esc);
    std::vector<uint32_t> idx(n_esc);
    for (int64_t i = 0; i < n_esc; ++i) {
        idx[i] = (uint32_t) (esc[i] >> 40);
        const uint64_t p = esc[i] & ((uint64_t) 1 << 40) - 1;
        for (int b = 0; b < 5; ++b) {
            patt[5*i + b] = (uint8_t) (p >> (8*b));
        }
    }

    // a code-767 super-block with an empty list is corrupt input; the dummy keeps the
    // binary search from dereferencing null instead of failing silently on the host
    uint8_t * esc_patt = nullptr;
    uint32_t * esc_idx = nullptr;
    CUDA_CHECK(cudaMalloc((void **) &esc_patt, std::max<size_t>(5, patt.size())));
    CUDA_CHECK(cudaMalloc((void **) &esc_idx,  std::max<size_t>(1, idx.size()*sizeof(uint32_t))));
    if (!patt.empty()) {
        CUDA_CHECK(cudaMemcpy(esc_patt, patt.data(), patt.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(esc_idx, idx.data(), idx.size()*sizeof(uint32_t), cudaMemcpyHostToDevice));
    }
    t.esc_patt = esc_patt;
    t.esc_idx  = esc_idx;

    std::lock_guard<std::mutex> lock(mutex);
    cache.emplace(key, t);
    return t;
}
