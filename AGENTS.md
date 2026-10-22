This is a fork for llama.cpp, intended for personal use only, and is intended to run a specific model.

The model is at `../models`, a Qwen35 architecture model. There are two variants currently, and we default to PZL, which only has Chinese + English token embeddings and pruned out others.

There is also a `.md` file describing its architecture. The model is launched by `~/.local/bin/llama-bonsai.sh` on an 8 GB RTX 4060 with 131,072 context length (quantized as turbo3), so it fits in and VRAM headroom is extremely tight.

Our Python venv is at `~/tamper/.venv`.
