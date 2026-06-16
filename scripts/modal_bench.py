"""
scripts/modal_bench.py — Run the leaderboard benchmark on Modal (GPU).

Presets (pick one):
    modal run scripts/modal_bench.py --preset smoke     # ~$0.50, 2 min  — sanity check
    modal run scripts/modal_bench.py --preset quick      # ~$2,   10 min — iterate fast
    modal run scripts/modal_bench.py --preset standard   # ~$25,  2h     — official benchmark
    modal run scripts/modal_bench.py --preset full        # ~$75,  6h     — publication-grade

Custom:
    modal run scripts/modal_bench.py --method sfp --memory 128 --preset quick
"""

from __future__ import annotations

from pathlib import Path

import modal

ROOT_PATH = Path(__file__).parent.parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.0",
        "transformers>=4.40",
        "peft>=0.10",
        "datasets>=2.18",
        "accelerate>=0.28",
        "numpy>=1.24",
        "tqdm>=4.66",
        "pyyaml>=6.0",
        "scipy",
        "prometheus-client>=0.20",
    )
    .add_local_dir(ROOT_PATH, remote_path="/root/sfp", ignore=[
        ".git/", ".venv/", "__pycache__/", "paper/", "runs/",
        ".ruff_cache/", ".pytest_cache/", "*.pdf",
    ])
)

app = modal.App("sfp-benchmark", image=image)

PUSHGATEWAY = ""

# ---------------------------------------------------------------------------
# Presets: name -> (gpu, seeds, steps_per_task, model, tasks, est_cost)
# Costs estimated at: T4 ~$0.59/hr, L4 ~$0.80/hr, A10G ~$1.10/hr
# ---------------------------------------------------------------------------

PRESETS = {
    "smoke": {
        "gpu": "T4",
        "seeds": [42],
        "steps_per_task": 50,
        "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "tasks": "math,code",
        "est_cost": "$0.50",
        "est_time": "~2 min",
        "desc": "Sanity check. Tiny model, 1 seed, 50 steps. Just verifies your method runs.",
    },
    "quick": {
        # A10G (24 GiB) instead of T4 (14.5 GiB) because distill-SETUP methods
        # need to keep teacher + student of Qwen-1.5B in VRAM simultaneously
        # — that's ~13.5 GiB before activations, OOMs a T4 on the first batch.
        "gpu": "A10G",
        "seeds": [42],
        "steps_per_task": 200,
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "tasks": "math,code,ifeval",
        "est_cost": "$2",
        "est_time": "~10 min",
        "desc": "Fast iteration. Real model, 1 seed, 200 steps. Good for development.",
    },
    "standard": {
        "gpu": "A10G",
        # Reduced from [42, 43, 44] to [42] to fit a ~$27 self-serve budget on
        # L40S (3 seeds × ~7h × $1.95 ≈ $40, 1 seed ≈ $14). Single-seed point
        # estimate without variance bars, but real eval (200 samples × 5 tasks)
        # so far less noisy than smoke. Paradigm's CI keeps all 3 seeds.
        "seeds": [42],
        "steps_per_task": 2000,
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "tasks": "math,code,ifeval,safety,domain",
        "est_cost": "$25",
        "est_time": "~2h",
        "desc": "Official leaderboard benchmark. 3 seeds, 2000 steps, 5 tasks. Used for PR scoring.",
    },
    "full": {
        "gpu": "A10G",
        "seeds": [42, 43, 44],
        "steps_per_task": 2000,
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "tasks": "math,code,ifeval,safety,domain",
        "est_cost": "$75",
        "est_time": "~6h",
        "desc": "Publication-grade benchmark. 3 seeds, 3 orderings, 2000 steps, 5 tasks.",
    },
}


@app.function(gpu="T4", timeout=7200)
def run_seed_t4(
    method: str, memory: int, seed: int, steps: int, model_name: str, tasks: str,
    code: str = "",
) -> dict:
    """Run benchmark on T4."""
    return _run(method, memory, seed, steps, model_name, tasks, code)


@app.function(gpu="L40S", timeout=7200)
def run_seed_a10g(
    method: str, memory: int, seed: int, steps: int, model_name: str, tasks: str,
    code: str = "",
) -> dict:
    """Run benchmark on L40S (48 GiB). Name kept for compat — A10G's 22 GiB
    is just barely too small for distill-SETUP methods on Qwen2.5-1.5B
    (teacher + student + activations + per-example KL intermediates need
    a hair over 22 GiB)."""
    return _run(method, memory, seed, steps, model_name, tasks, code)


def _run(method: str, memory: int, seed: int, steps: int, model_name: str, tasks: str, code: str = "") -> dict:
    """Shared training logic. Pushes metrics to Prometheus on dev-georgios."""
    import json
    import subprocess

    out_dir = f"/root/sfp/out/bench/{method}_m{memory}_s{seed}"

    cmd = [
        "python", "-u", "train.py",  # -u: unbuffered stdout so progress streams to Modal logs
        "--method", method,
        "--memory", str(memory),
        "--seed", str(seed),
        "--model", model_name,
        "--tasks", tasks,
        "--steps_per_task", str(steps),
        "--lora_rank", "64",
        "--out_dir", out_dir,
        # Use evaluate.py's "smoke" eval (few samples per task) for both smoke
        # AND quick presets — on T4, Qwen-1.5B does ~11s/sample on GSM8K
        # generation, so 200-sample "fast" eval makes quick take 5-7 hours
        # instead of the documented ~10 min. "standard"/"full" still use fast.
        "--mode", "smoke" if steps <= 200 else "fast",
        "--pushgateway", PUSHGATEWAY,
    ]

    # If code is provided, write to a temp file and pass --method-file
    if code:
        method_file = f"/tmp/method_{method}.py"
        with open(method_file, "w") as f:
            f.write(code)
        cmd.extend(["--method_file", method_file])

    # Workaround for stale HF dataset revision pins in data.py (mbpp
    # 12e9221 etc. now return 404 from the Hub). Write a sitecustomize.py
    # that monkey-patches datasets.load_dataset to drop the revision kwarg,
    # then put it on PYTHONPATH so Python auto-loads it before data.py
    # imports datasets. No touch to data.py itself.
    import os as _os
    patch_dir = "/tmp/sfp_patches"
    _os.makedirs(patch_dir, exist_ok=True)
    with open(f"{patch_dir}/sitecustomize.py", "w") as _f:
        _f.write(
            "import datasets as _d\n"
            "_o = _d.load_dataset\n"
            "def _p(*a, **k):\n"
            "    k.pop('revision', None)\n"
            "    return _o(*a, **k)\n"
            "_d.load_dataset = _p\n"
        )
    env = dict(_os.environ)
    env["PYTHONPATH"] = f"{patch_dir}:{env.get('PYTHONPATH', '')}"
    # Help PyTorch handle fragmented memory; matters when teacher + student
    # of a 1.5B model + per-step softmax over a 151k vocab leave little
    # headroom. Modal's own OOM message suggested this exact setting.
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Stream subprocess output directly to Modal App Logs (don't capture).
    # The previous capture_output=True hid all train.py progress until the
    # subprocess finished, leaving Modal logs blank for the entire ~10 min run.
    result = subprocess.run(cmd, cwd="/root/sfp", env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Training failed (seed={seed}, returncode={result.returncode}) — see Modal logs above for stderr.")

    results_path = f"{out_dir}/results.json"
    with open(results_path) as f:
        return json.load(f)


@app.local_entrypoint()
def main(method: str = "naive", memory: int = 128, preset: str = "quick", code: str = ""):
    """Run benchmark with cost-aware presets."""
    import json
    import statistics

    if preset not in PRESETS:
        print(f"Unknown preset: {preset}")
        print(f"Available: {', '.join(PRESETS.keys())}")
        raise SystemExit(1)

    p = PRESETS[preset]
    seeds = p["seeds"]
    steps = p["steps_per_task"]
    model_name = p["model"]
    tasks = p["tasks"]
    gpu = p["gpu"]

    print("=" * 60)
    print("  SFP Leaderboard Benchmark")
    print("=" * 60)
    print(f"  Preset     : {preset} — {p['desc']}")
    print(f"  Est. cost  : {p['est_cost']}")
    print(f"  Est. time  : {p['est_time']}")
    print(f"  GPU        : {gpu}")
    print(f"  Method     : {method}")
    print(f"  Memory     : {memory}")
    print(f"  Model      : {model_name}")
    print(f"  Tasks      : {tasks}")
    print(f"  Steps/task : {steps}")
    print(f"  Seeds      : {seeds}")
    if code:
        print(f"  Code       : {len(code)} bytes (submitted)")
    print("=" * 60)
    print()

    # Pick the right GPU function
    run_fn = run_seed_a10g if gpu == "A10G" else run_seed_t4

    # Run all seeds in parallel on Modal
    results = list(run_fn.map(
        [method] * len(seeds),
        [memory] * len(seeds),
        seeds,
        [steps] * len(seeds),
        [model_name] * len(seeds),
        [tasks] * len(seeds),
        [code] * len(seeds),
    ))

    # Aggregate
    retentions = [r.get("retention", 0.0) for r in results]
    plasticities = [r.get("plasticity", 0.0) for r in results]
    scores = [r.get("score", 0.0) for r in results]

    ret_mean = statistics.mean(retentions)
    pla_mean = statistics.mean(plasticities)
    sco_mean = statistics.mean(scores)
    ret_std = statistics.stdev(retentions) if len(retentions) > 1 else 0.0
    pla_std = statistics.stdev(plasticities) if len(plasticities) > 1 else 0.0
    sco_std = statistics.stdev(scores) if len(scores) > 1 else 0.0

    print(f"\n{'=' * 60}")
    print("  RESULTS")
    print(f"{'=' * 60}")
    print(f"  Preset     : {preset}")
    print(f"  Method     : {method}")
    print(f"  Memory     : {memory}")
    print(f"  Retention  : {ret_mean:.4f} +/- {ret_std:.4f}")
    print(f"  Plasticity : {pla_mean:.4f} +/- {pla_std:.4f}")
    print(f"  Score      : {sco_mean:.4f} +/- {sco_std:.4f}")
    if preset != "standard":
        print(f"\n  NOTE: This is a '{preset}' run. For official scores, use --preset standard")
    print(f"{'=' * 60}\n")

    # Write machine-readable output
    output = {
        "method": method,
        "memory": memory,
        "preset": preset,
        "gpu": gpu,
        "retention_mean": round(ret_mean, 4),
        "retention_std": round(ret_std, 4),
        "plasticity_mean": round(pla_mean, 4),
        "plasticity_std": round(pla_std, 4),
        "score_mean": round(sco_mean, 4),
        "score_std": round(sco_std, 4),
        "per_seed": results,
    }
    out_path = Path("benchmark_results.json")
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"Results written to {out_path}")
