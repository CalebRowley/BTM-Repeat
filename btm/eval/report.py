import argparse
import csv
import json
from pathlib import Path

import numpy as np

from btm.ensemble import ensemble_from_arrays, perplexity, posterior_entropy
from btm.posterior import cached_prior, sequence_loglik
from btm.utils import git_sha, load_config

DEVIATIONS = [
    "Seed: a public pretrained seed (Pythia-160M-deduped, ~300B tokens) replaces BTM's own seed phase. "
    "There is no seed-fraction knob, and the seed has seen Pile domains, including GitHub and FreeLaw.",
    "Optimizer: each ELM starts with fresh AdamW and its own warmup. The paper continued the seed's optimizer state.",
    "Domains and iterations: 3 domains and 2 BTM iterations. The seed is included as a candidate when averaging for branching.",
    "Branching weights: the mean of per-window posteriors (uniform prior) over 100 dev windows.",
    "Law data: TeraflopAI/Caselaw-Access-Project replaces free-law/Caselaw_Access_Project (gated with manual approval).",
    "Eval data: some evaluation data may overlap with the seed's pretraining data.",
    "No compute-matched dense baseline (out of scope).",
]

CONTAMINATION = (
    "The seed was trained on the Pile, which includes GitHub and FreeLaw, so some test documents may already "
    "have been seen by the seed. All models share the seed, so this inflates absolute quality but barely "
    "affects comparisons between them. No attempt was made to remove the overlap."
)


class LogProbs:
    def __init__(self, root):
        self.root = Path(root)
        self.cache = {}

    def get(self, model, domain, split):
        key = (model, domain, split)
        if key not in self.cache:
            path = self.root / model / f"{domain}_{split}.npy"
            self.cache[key] = np.load(path) if path.exists() else None
        return self.cache[key]


def fmt(x, digits=3):
    return "n/a" if x is None else f"{x:.{digits}f}"


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def macro(values):
    return None if not values or any(v is None for v in values) else float(np.mean(values))


def load_metrics(path):
    by_key = {}
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        if r.get("event") in ("train", "eval"):
            by_key[(r["event"], r["step"])] = r
    train = sorted((r for (e, _), r in by_key.items() if e == "train"), key=lambda r: r["step"])
    evals = sorted((r for (e, _), r in by_key.items() if e == "eval"), key=lambda r: r["step"])
    return train, evals


def plot_curves(log_root, runs, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []
    for run in runs:
        path = Path(log_root) / run / "metrics.jsonl"
        if not path.exists():
            continue
        train, evals = load_metrics(path)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot([r["step"] for r in train], [r["loss"] for r in train], alpha=0.5, label="train")
        ax.plot([r["step"] for r in evals], [r["dev_loss_own"] for r in evals], marker="o", label="dev (own domain)")
        ax.plot([r["step"] for r in evals], [r["dev_loss_web"] for r in evals], marker="o", label="dev (web)")
        ax.set_xlabel("step")
        ax.set_ylabel("loss (nats/token)")
        ax.set_title(run)
        ax.legend()
        fig.tight_layout()
        out = Path(out_dir) / f"curve_{run}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        written.append(out.name)
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    lp = LogProbs(cfg["logprob_root"])
    out_dir = Path(cfg["results_root"])
    out_dir.mkdir(parents=True, exist_ok=True)
    domains, in_domain, held_out = cfg["domains"], cfg["in_domain"], cfg["held_out"]
    elms, n_prior = cfg["elms"], int(cfg["prior_windows"])

    priors = {}
    for d in domains:
        devs = [lp.get(e, d, "dev") for e in elms]
        if all(x is not None for x in devs):
            priors[d] = cached_prior(np.stack([sequence_loglik(x[:n_prior]) for x in devs], 1))

    rows = {}

    def single(label, model):
        rows[label] = {d: None if lp.get(model, d, "test") is None else perplexity(lp.get(model, d, "test")) for d in domains}

    single("M0 seed", "seed")
    for i, e in enumerate(elms, 1):
        single(f"M{i} {e}", e)

    ens_labels = ["E1 ensemble (cached prior)", "E2 ensemble (uniform prior)"] + [f"E3 ensemble (top-{k})" for k in cfg["topk"]]
    for label in ens_labels:
        rows[label] = {d: None for d in domains}
    entropy = {}
    for d in domains:
        tests = [lp.get(e, d, "test") for e in elms]
        if any(x is None for x in tests):
            continue
        rows["E2 ensemble (uniform prior)"][d] = perplexity(ensemble_from_arrays(tests, np.full(len(elms), 1 / len(elms)))[0])
        if d not in priors:
            continue
        lp_ens, logpost = ensemble_from_arrays(tests, priors[d])
        rows["E1 ensemble (cached prior)"][d] = perplexity(lp_ens)
        final = np.exp(logpost[:, :, -1])
        entropy[d] = {
            "mean_token_entropy": float(posterior_entropy(logpost).mean()),
            "final_posterior_mean": final.mean(1).tolist(),
            "final_max_posterior_p10_p50_p90": np.percentile(final.max(0), [10, 50, 90]).tolist(),
            "frac_windows_final_max_gt_0.99": float((final.max(0) > 0.99).mean()),
        }
        for k in cfg["topk"]:
            rows[f"E3 ensemble (top-{k})"][d] = perplexity(ensemble_from_arrays(tests, priors[d], k=k)[0])

    single("A1 param avg (uniform)", "avg_uniform")
    rows["A2 param avg (posterior)"] = {
        d: None if lp.get(f"avg_post_{d}", d, "test") is None else perplexity(lp.get(f"avg_post_{d}", d, "test"))
        for d in domains
    }

    header = ["model", *domains, "macro in-domain", "macro held-out"]
    table = []
    for label, vals in rows.items():
        table.append([label, *[vals[d] for d in domains], macro([vals[d] for d in in_domain]), macro([vals[d] for d in held_out])])
    write_csv(out_dir / "ppl_test.csv", header, table)

    seed_rows = [
        [d, *[fmt(None if lp.get("seed", d, s) is None else perplexity(lp.get("seed", d, s))) for s in ("dev", "test")]]
        for d in domains
    ]
    write_csv(out_dir / "seed_ppl.csv", ["domain", "dev", "test"], seed_rows)

    prior_rows = [[d, *[fmt(p, 4) for p in priors[d]]] for d in domains if d in priors]
    write_csv(out_dir / "priors.csv", ["domain", *elms], prior_rows)

    ent_rows = [[d, fmt(v["mean_token_entropy"], 4), *[fmt(p, 4) for p in v["final_posterior_mean"]],
                 *[fmt(p, 4) for p in v["final_max_posterior_p10_p50_p90"]], fmt(v["frac_windows_final_max_gt_0.99"], 3)]
                for d, v in entropy.items()]
    ent_header = ["domain", "mean_token_entropy", *[f"final_mean_{e}" for e in elms], "final_max_p10", "final_max_p50", "final_max_p90", "frac_final_max_gt_0.99"]
    write_csv(out_dir / "entropy.csv", ent_header, ent_rows)

    flags = []
    for d, e in zip(in_domain, elms):
        e1, s = rows["E1 ensemble (cached prior)"][d], rows[f"M{elms.index(e) + 1} {e}"][d]
        if e1 is not None and s is not None and e1 > s * 1.01:
            flags.append(f"E1 on {d} test ({e1:.3f}) is more than 1% worse than {e} ({s:.3f}).")

    gate_b = []
    law_init_weights = Path(cfg["ckpt_root"]) / "elm_law_init" / "weights.json"
    for label, model in [("seed", "seed"), ("uniform avg of candidates", "law_init_uniform"), ("posterior-weighted init", "elm_law_init")]:
        arr = lp.get(model, "law", "dev")
        gate_b.append([label, fmt(None if arr is None else float(-arr.mean(dtype=np.float64)), 4)])
    if gate_b[0][1] != "n/a" and gate_b[2][1] != "n/a" and float(gate_b[2][1]) > 1.05 * float(gate_b[0][1]):
        flags.append("Posterior-weighted law init dev NLL is more than 5% worse than the seed.")

    curves = plot_curves(cfg["log_root"], elms, out_dir)

    md = ["# BTM baseline report", "", f"git SHA: `{git_sha()}`", ""]
    md += ["## 1. Test perplexity", "", md_table(header, [[r[0], *[fmt(v) for v in r[1:]]] for r in table]), ""]
    md += ["Seed perplexity (dev / test):", "", md_table(["domain", "dev", "test"], seed_rows), ""]
    md += ["## 2. Cached prior weights (first %d dev windows)" % n_prior, "", md_table(["domain", *elms], prior_rows), ""]
    md += ["## 3. Law initialization", ""]
    if law_init_weights.exists():
        w = json.loads(law_init_weights.read_text())
        md += [md_table(["candidate", "weight"], [[p, fmt(x, 4)] for p, x in zip(w["paths"], w["weights"])]), ""]
    md += ["Law dev NLL (Gate B):", "", md_table(["model", "law dev NLL"], gate_b), ""]
    md += ["## 4. Posterior entropy under E1 (test)", "", md_table(ent_header, ent_rows), ""]
    md += ["## 5. Training curves", ""] + [f"![{c}]({c})" for c in curves] + [""]
    md += ["## 6. Deviations from the paper", ""] + [f"- {d}" for d in DEVIATIONS] + ["", "**Contamination note.** " + CONTAMINATION, ""]
    md += ["## Flags", ""] + ([f"- {f}" for f in flags] or ["- none"]) + [""]
    (out_dir / "REPORT.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
