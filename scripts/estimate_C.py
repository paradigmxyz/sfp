"""Estimate forgetting constant C_ℓ at each layer of a model.

Usage:
    python scripts/estimate_C.py --model HuggingFaceTB/SmolLM2-135M-Instruct --memory 256

Runs a mini fine-tuning loop (1 task), collects pre/post activations at every
transformer layer, and estimates C_ℓ = Cov(Δloss, ‖P_W(Δa)‖) / Var(‖P_W(Δa)‖).

Outputs a table of C_ℓ per layer, sorted by ascending |C_ℓ| (best layers first).
"""
import argparse
import torch
from model import load_model
from features import (get_layer_names, collect_activations, build_pca_basis,
                      select_top_r, estimate_forgetting_constant, davis_kahan_buffer_size)
from data import load_task_data


def main():
    parser = argparse.ArgumentParser(description="Estimate forgetting constant C per layer")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--memory", type=int, default=256, help="Memory buffer size")
    parser.add_argument("--k", type=int, default=128, help="PCA components")
    parser.add_argument("--r", type=int, default=32, help="Preserved dimensions")
    parser.add_argument("--steps", type=int, default=50, help="Fine-tuning steps for estimation")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model, tokenizer = load_model(args.model)
    device = next(model.parameters()).device

    # Probe all layers at fine granularity
    n_probe = 20
    positions = [i / n_probe for i in range(1, n_probe)]
    layer_names = get_layer_names(model, positions=positions)
    print(f"Probing {len(layer_names)} layers: {layer_names}")

    # Load first task data as "old task"
    print("Loading task data...")
    task_data = load_task_data("math", tokenizer, max_samples=args.memory * 2)
    mem_samples = task_data[:args.memory]

    # Collect pre-training activations
    print("Collecting pre-training activations...")
    pre_acts = collect_activations(model, mem_samples, tokenizer, layer_names)

    # Compute per-sample old-task loss before training
    print("Computing pre-training losses...")
    pre_losses = []
    model.eval()
    with torch.no_grad():
        for sample in mem_samples:
            text = sample.get("input", sample.get("text", ""))
            enc = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=512).to(device)
            enc["labels"] = enc["input_ids"].clone()
            loss = model(**enc).loss.item()
            pre_losses.append(loss)
    pre_losses = torch.tensor(pre_losses)

    # Fine-tune on "new task" (code) for a few steps
    print(f"Fine-tuning for {args.steps} steps on code task...")
    new_task_data = load_task_data("code", tokenizer, max_samples=args.steps * 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    for i, sample in enumerate(new_task_data[:args.steps]):
        text = sample.get("input", sample.get("text", ""))
        enc = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=512).to(device)
        enc["labels"] = enc["input_ids"].clone()
        loss = model(**enc).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if (i + 1) % 10 == 0:
            print(f"  Step {i+1}/{args.steps}, loss={loss.item():.4f}")

    # Collect post-training activations and losses
    print("Collecting post-training activations...")
    post_acts = collect_activations(model, mem_samples, tokenizer, layer_names)

    print("Computing post-training losses...")
    post_losses = []
    model.eval()
    with torch.no_grad():
        for sample in mem_samples:
            text = sample.get("input", sample.get("text", ""))
            enc = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=512).to(device)
            enc["labels"] = enc["input_ids"].clone()
            loss = model(**enc).loss.item()
            post_losses.append(loss)
    post_losses = torch.tensor(post_losses)

    forgetting = post_losses - pre_losses  # [n_mem]

    # Estimate C_ℓ at each layer
    print("\n" + "=" * 70)
    print(f"{'Layer':<30} {'C_ℓ':>10} {'SE':>10} {'Δ_k':>10} {'n_DK':>8} {'VarRatio':>10}")
    print("=" * 70)

    results = []
    for layer_name in layer_names:
        pre_a = pre_acts[layer_name]
        post_a = post_acts[layer_name]
        delta_a = post_a - pre_a  # [n, hidden]

        u, explained_var = build_pca_basis(pre_a, k=args.k)
        u_r = select_top_r(u, explained_var, r=args.r)

        # Subspace drift magnitude per sample
        drift = torch.norm(delta_a @ u_r, dim=1)  # [n]

        C_hat, std_err = estimate_forgetting_constant(drift, forgetting)

        # Eigenvalue gap
        if len(explained_var) > args.r:
            delta_k = (explained_var[args.r - 1] - explained_var[args.r]).item()
        else:
            delta_k = explained_var[-1].item()

        n_dk = davis_kahan_buffer_size(delta_k=max(delta_k, 1e-8), sigma=1.0)

        var_ratio = explained_var[:args.r].sum().item() / max(explained_var.sum().item(), 1e-12)

        results.append({
            "layer": layer_name, "C": C_hat, "SE": std_err,
            "delta_k": delta_k, "n_dk": n_dk, "var_ratio": var_ratio,
        })
        print(f"{layer_name:<30} {C_hat:>10.4f} {std_err:>10.4f} {delta_k:>10.6f} {n_dk:>8d} {var_ratio:>10.4f}")

    # Sort by |C| ascending (smallest = tightest bound)
    results.sort(key=lambda x: abs(x["C"]) if x["C"] != float("inf") else 1e9)

    print("\n" + "=" * 70)
    print("TOP 3 LAYERS (smallest |C_ℓ| = tightest forgetting bound):")
    print("=" * 70)
    for r in results[:3]:
        print(f"  {r['layer']:<30} C={r['C']:.4f}  (Davis-Kahan n_min={r['n_dk']})")

    print("\nBOTTOM 3 LAYERS (largest |C_ℓ| = loosest bound):")
    for r in results[-3:]:
        print(f"  {r['layer']:<30} C={r['C']:.4f}")


if __name__ == "__main__":
    main()
