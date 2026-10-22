#!/usr/bin/env python3
"""Benchmark llama-server speculative-decoding configurations.

Runs one llama-server per configuration, drives a workload against it, and
reports effective generation throughput together with the per-implementation
draft statistics that the server prints at trace verbosity.

Two workloads:

  rewrite  Deterministic. Concatenates source files into one prompt and asks the
           model to regerenerate the target file. Cheap (~2 min) and repeatable,
           so it is the right tool for sweeping drafter parameters. It is an
           optimistic workload: the model recites long verbatim runs.

  agent    Real agentic session. Runs an external agent CLI (for example
           `pii -p "$SPEC_BENCH_TASK"`) inside a fixture tree, resetting the
           tree before every run. Slow (minutes) and noisy, because every run
           is a different sampled trajectory, but it is the workload that
           actually matters.

Examples:

  # sweep ngram parameters on the deterministic workload
  ./spec_bench.py -m model.gguf \\
      --config 'none=--spec-type none' \\
      --config 'mapk48=--spec-type ngram-map-k' \\
      --config 'mapk16=--spec-type ngram-map-k --spec-ngram-map-k-size-m 16'

  # try chained drafters (the first one that produces a draft wins)
  ./spec_bench.py -m model.gguf \\
      --config 'mapk=--spec-type ngram-map-k' \\
      --config 'mapk+suffix=--spec-type ngram-map-k,ngram-suffix --spec-ngram-suffix-n-max 48 --spec-ngram-suffix-max-factor 3 --spec-ngram-suffix-max-depth 32'

  # real agent run on a fixture repo
  ./spec_bench.py -m model.gguf --mode agent \\
      --agent-dir /tmp/fixture --agent-cmd 'pii -p "$SPEC_BENCH_TASK" --no-session' \\
      --config 'none=--spec-type none' --config 'mapk=--spec-type ngram-map-k'

Notes:
  * The server runs with --verbosity 4 so that `statistics <impl>:` lines and the
    per-position acceptance rates are logged. Those lines are what tells you
    which drafter fired and how deep its drafts stay useful.
  * Metrics are taken from the server's own timings, not from wall clock, so
    token throughput is comparable across configurations even when the agent
    takes a different number of turns.
"""

import argparse
import glob
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


# --------------------------------------------------------------------------- #
# log parsing
# --------------------------------------------------------------------------- #

RE_EVAL      = re.compile(r'\beval time =\s+([\d.]+) ms /\s+(\d+) tokens')
RE_PROMPT    = re.compile(r'prompt eval time =\s+([\d.]+) ms /\s+(\d+) tokens')
RE_ACCEPT    = re.compile(r'draft acceptance = ([\d.]+) \(\s*(\d+) accepted /\s*(\d+) generated\), mean len =\s*([\d.]+)')
RE_ACCEPTPOS = re.compile(r'acc per pos = \(([^)]*)\)')
RE_IMPLSTAT  = re.compile(r'statistics\s+(\S+): #calls\(b,g,a\) =(.*?)#gen tokens =\s*(\d+), #acc tokens =\s*(\d+)(.*)')
RE_IMPLPOS   = re.compile(r'#acc rate/pos = \(([^)]*)\)')
RE_ADDED     = re.compile(r"adding speculative implementation '([^']+)'")


def parse_log(path):
    """Extract the metrics that matter from one llama-server log."""
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()

    gen_ms = gen_tok = 0.0
    prompt_ms = prompt_tok = 0.0
    n_rounds = 0.0          # number of verify rounds (drafts verified)
    acc = gen = 0           # accepted / generated draft tokens
    pos_acc = [0.0] * 64    # per draft position acceptance (averaged)
    n_pos_lines = 0

    for line in lines:
        if "prompt eval time" in line:
            m = RE_PROMPT.search(line)
            if m:
                prompt_ms += float(m.group(1))
                prompt_tok += int(m.group(2))
            continue
        m = RE_EVAL.search(line)
        if m:
            gen_ms += float(m.group(1))
            gen_tok += int(m.group(2))
            continue
        m = RE_ACCEPT.search(line)
        if m:
            a, g, mean_len = int(m.group(2)), int(m.group(3)), float(m.group(4))
            acc += a
            gen += g
            if mean_len > 1.0 and a > 0:
                n_rounds += a / (mean_len - 1.0)
            continue
        m = RE_ACCEPTPOS.search(line)
        if m:
            vals = [float(x) for x in m.group(1).split(",") if x.strip()]
            for i, v in enumerate(vals[:len(pos_acc)]):
                pos_acc[i] += v
            n_pos_lines += 1

    # the per-implementation counters are cumulative over the server lifetime,
    # so the last snapshot is the total (summing the snapshots would over-count)
    impls = {}
    for line in lines:
        m = RE_IMPLSTAT.search(line)
        if not m:
            continue
        name = m.group(1)
        ent = impls.setdefault(name, {"gen": 0, "acc": 0})
        ent["gen"] = max(ent["gen"], int(m.group(3)))
        ent["acc"] = max(ent["acc"], int(m.group(4)))
        pm = RE_IMPLPOS.search(m.group(5))
        if pm:
            vals = [float(x) for x in pm.group(1).split(",") if x.strip()]
            if vals:
                ent["acc_per_pos"] = vals

    return {
        "gen_tokens": int(gen_tok),
        "gen_ms": round(gen_ms, 1),
        "tok_s": round(gen_tok * 1000.0 / gen_ms, 2) if gen_ms else None,
        "prompt_tokens": int(prompt_tok),
        "prompt_ms": round(prompt_ms, 1),
        "draft_gen": gen,
        "draft_acc": acc,
        "acc_rate": round(acc / gen, 4) if gen else None,
        "rounds": int(round(n_rounds)),
        "tok_per_round": round(1.0 + acc / n_rounds, 2) if n_rounds else None,
        "acc_per_pos": ([round(v / n_pos_lines, 3) for v in pos_acc] if n_pos_lines else []),
        "impls": impls,
        "added_impls": [l.group(1) for l in (RE_ADDED.search(x) for x in lines) if l],
    }


def merge_logs(paths):
    """Aggregate several server logs (e.g. repeated runs) into one summary."""
    parts = [parse_log(p) for p in paths]
    out = {"runs": len(parts), "gen_tokens": sum(p["gen_tokens"] for p in parts),
           "gen_ms": sum(p["gen_ms"] for p in parts),
           "prompt_tokens": sum(p["prompt_tokens"] for p in parts),
           "draft_gen": sum(p["draft_gen"] for p in parts),
           "draft_acc": sum(p["draft_acc"] for p in parts),
           "rounds": sum(p["rounds"] for p in parts)}
    out["tok_s"] = round(out["gen_tokens"] * 1000.0 / out["gen_ms"], 2) if out["gen_ms"] else None
    out["acc_rate"] = round(out["draft_acc"] / out["draft_gen"], 4) if out["draft_gen"] else None
    out["tok_per_round"] = round(1.0 + out["draft_acc"] / out["rounds"], 2) if out["rounds"] else None
    out["acc_per_pos"] = parts[-1]["acc_per_pos"] if parts else []
    out["impls"] = {}
    for p in parts:
        for name, ent in p["impls"].items():
            dst = out["impls"].setdefault(name, {"gen": 0, "acc": 0})
            dst["gen"] = max(dst["gen"], ent["gen"])
            dst["acc"] = max(dst["acc"], ent["acc"])
            if "acc_per_pos" in ent:
                dst["acc_per_pos"] = ent["acc_per_pos"]
    return out


# --------------------------------------------------------------------------- #
# workloads
# --------------------------------------------------------------------------- #

def build_rewrite_prompt(args):
    files = []
    for pattern in args.files:
        hit = sorted(glob.glob(os.path.join(args.repo_root, pattern)))
        files.extend(hit)
    if not files:
        sys.exit("error: --files matched nothing under %s" % args.repo_root)

    target = args.target
    if target is None:
        target = os.path.relpath(files[-1], args.repo_root)

    parts = ["Repo context below.\n"]
    for f in files:
        rel = os.path.relpath(f, args.repo_root)
        with open(f, encoding="utf-8", errors="replace") as fh:
            parts.append("\n### %s\n```\n%s\n```\n" % (rel, fh.read()))
    parts.append("\n### task\n" + args.instruction.format(target=target) +
                 "\nOutput the complete file in one code block, nothing else.\n")
    return "".join(parts)


def run_rewrite(port, args):
    prompt = args.prompt if args.prompt else build_rewrite_prompt(args)
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.n_predict,
        "temperature": args.temp,
        "top_k": 1,
        "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": args.thinking},
    }
    t0 = time.time()
    res = http_json("http://127.0.0.1:%d/v1/chat/completions" % port, body, args.timeout)
    wall = time.time() - t0
    out = ""
    try:
        out = res["choices"][0]["message"].get("content") or ""
    except Exception:
        pass
    return {"wall_s": round(wall, 1), "reply_chars": len(out)}


def run_agent(port, args):
    if not args.agent_cmd:
        sys.exit("error: --mode agent needs --agent-cmd")
    env = dict(os.environ)
    env["SPEC_BENCH_TASK"] = args.task
    env["SPEC_BENCH_URL"] = "http://127.0.0.1:%d" % port
    if args.agent_reset:
        subprocess.run(args.agent_reset, shell=True, cwd=args.agent_dir, check=False)
    t0 = time.time()
    proc = subprocess.run(args.agent_cmd, shell=True, cwd=args.agent_dir, env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    wall = time.time() - t0
    if args.keep_logs:
        with open(os.path.join(args.out_dir, "agent_stdout.log"), "wb") as fh:
            fh.write(proc.stdout or b"")
    return {"wall_s": round(wall, 1), "agent_rc": proc.returncode,
            "agent_tail": (proc.stdout or b"").decode("utf-8", "replace").strip().splitlines()[-3:]}


# --------------------------------------------------------------------------- #
# server control
# --------------------------------------------------------------------------- #

def http_json(url, body, timeout):
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def wait_healthy(port, proc, timeout=300):
    deadline = time.time() + timeout
    url = "http://127.0.0.1:%d/health" % port
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if json.loads(r.read().decode()).get("status") == "ok":
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def server_args(args, cfg_flags):
    cmd = [args.server, "-m", args.model, "-a", "bench",
           "--host", "127.0.0.1", "--port", str(args.port),
           "-ngl", str(args.ngl), "-fa", args.flash_attn,
           "-c", str(args.ctx), "-ctk", args.kv, "-ctv", args.kv,
           "-np", "1", "-ub", str(args.ub), "-b", str(args.batch),
           "--jinja", "--verbosity", str(args.verbosity)]
    cmd += shlex.split(args.extra or "")
    cmd += cfg_flags
    return cmd


def run_config(name, cfg_flags, args):
    log_path = os.path.join(args.out_dir, "%s_%d.log" % (name, args.port))
    if os.path.exists(log_path):
        os.remove(log_path)
    cmd = server_args(args, cfg_flags)
    print("  server: %s" % " ".join(shlex.quote(c) for c in cmd), flush=True)
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        try:
            if not wait_healthy(args.port, proc):
                proc.kill()
                sys.exit("error: server did not become healthy, see %s" % log_path)
            extra = run_rewrite(args.port, args) if args.mode == "rewrite" else run_agent(args.port, args)
        finally:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=60)      # let it print the tear-down statistics
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    stats = merge_logs([log_path])
    stats.update(extra)
    stats["wall_s"] = extra.get("wall_s")
    stats["config"] = name
    stats["flags"] = " ".join(shlex.quote(f) for f in cfg_flags)
    return stats


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #

def fmt_pos(acc_per_pos, limit=12):
    if not acc_per_pos:
        return "-"
    return " ".join("%.2f" % v for v in acc_per_pos[:limit])


def report(results, args):
    base = results[0]["tok_s"] if results and results[0].get("tok_s") else None
    print()
    print("| config | tok/s | vs %s | gen tokens | draft acc | tok/round | wall | impls (acc/gen) |" % results[0]["config"])
    print("|---|---|---|---|---|---|---|---|")
    for r in results:
        speed = r.get("tok_s")
        rel = ("%.2fx" % (speed / base)) if (speed and base) else "-"
        impls = ", ".join("%s %d/%d" % (k, v["acc"], v["gen"]) for k, v in sorted(r["impls"].items())) or "-"
        print("| %s | %s | %s | %d | %s | %s | %ss | %s |" % (
            r["config"], speed if speed is not None else "-", rel, r["gen_tokens"],
            ("%.3f" % r["acc_rate"]) if r.get("acc_rate") is not None else "-",
            r.get("tok_per_round", "-"), r.get("wall_s", "-"), impls))
    print()
    for r in results:
        if r.get("acc_per_pos"):
            print("acceptance per draft position, %s: %s" % (r["config"], fmt_pos(r["acc_per_pos"])))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=1)
        print("\nraw results -> %s" % args.json)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default=os.path.join(REPO_ROOT, "build/bin/llama-server"))
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("--config", action="append", default=[], metavar="NAME:FLAGS",
                    help="one benchmarked configuration, repeatable; FLAGS is passed to llama-server")
    ap.add_argument("--repeat", type=int, default=1, help="repeat every configuration N times (default 1)")

    ap.add_argument("--mode", choices=["rewrite", "agent"], default="rewrite")
    ap.add_argument("--n-predict", type=int, default=1600)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--thinking", dest="thinking", action="store_true", help="keep model thinking on (rewrite mode)")
    ap.add_argument("--no-thinking", dest="thinking", action="store_false")
    ap.set_defaults(thinking=False)

    # rewrite workload
    ap.add_argument("--repo-root", default=REPO_ROOT)
    ap.add_argument("--files", nargs="+",
                    default=["common/ngram-map.cpp", "common/suffix-tree.cpp", "common/common.h"],
                    help="files concatenated into the prompt (globs, relative to --repo-root)")
    ap.add_argument("--target", default=None, help="file the model has to regenerate (default: last of --files)")
    ap.add_argument("--instruction", default="Rewrite {target} with the same content, "
                                            "changing only the comment separators.")
    ap.add_argument("--prompt", default=None, help="use this prompt file instead of building one")

    # agent workload
    ap.add_argument("--agent-cmd", default=None, help="agent CLI to run, e.g. 'pii -p \"$SPEC_BENCH_TASK\" --no-session'")
    ap.add_argument("--agent-dir", default=None, help="working tree the agent works in")
    ap.add_argument("--agent-reset", default="git checkout -- . && git clean -fd",
                    help="command that restores the fixture tree before each run")
    ap.add_argument("--task", default="Refactor the code in this repository and verify it builds.")

    # server
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("--ctx", type=int, default=131072)
    ap.add_argument("--kv", default="turbo3")
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--flash-attn", default="on")
    ap.add_argument("--ub", type=int, default=64)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--verbosity", type=int, default=4)
    ap.add_argument("--extra", default="", help="extra llama-server flags shared by all configurations")
    ap.add_argument("--out-dir", default="/tmp/spec-bench")
    ap.add_argument("--json", default=None)
    ap.add_argument("--keep-logs", action="store_true")
    ap.add_argument("--timeout", type=int, default=3600)

    args = ap.parse_args()

    if not args.config:
        sys.exit("error: at least one --config NAME:FLAGS is required")
    os.makedirs(args.out_dir, exist_ok=True)

    configs = []
    for c in args.config:
        name, _, flags = c.partition("=")
        configs.append((name.strip(), shlex.split(flags.strip())))

    results = []
    for rep in range(args.repeat):
        for name, flags in configs:
            if args.repeat > 1:
                name = "%s#%d" % (name, rep + 1)
            print("== %s (%s mode) ==" % (name, args.mode), flush=True)
            results.append(run_config(name, flags, args))

    report(results, args)


if __name__ == "__main__":
    main()
