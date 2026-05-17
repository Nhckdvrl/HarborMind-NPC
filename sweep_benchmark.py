#!/usr/bin/env python3
import argparse
import ast
import csv
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt


def run_one(script, base_url, model, concurrency, requests):
    cmd = [
        "python", script,
        "--base-url", base_url,
        "--model", model,
        "--concurrency", str(concurrency),
        "--requests", str(requests),
    ]

    print(f"\n[RUN] concurrency={concurrency}, requests={requests}")
    p = subprocess.run(cmd, text=True, capture_output=True)

    if p.returncode != 0:
        print(p.stdout)
        print(p.stderr)
        raise RuntimeError(f"benchmark failed at concurrency={concurrency}")

    out = p.stdout.strip()
    print(out)

    result = ast.literal_eval(out.splitlines()[-1])
    result["concurrency"] = concurrency
    result["qps"] = result["requests"] / result["wall_time_s"]
    return result


def load_existing(csv_path):
    if not csv_path.exists():
        return []

    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "concurrency": int(r["concurrency"]),
                "requests": int(r["requests"]),
                "avg_latency_s": float(r["avg_latency_s"]),
                "wall_time_s": float(r["wall_time_s"]),
                "tokens_per_second": float(r["tokens_per_second"]),
                "qps": float(r["qps"]),
            })
    return rows


def save_csv(csv_path, rows):
    keys = [
        "concurrency",
        "requests",
        "avg_latency_s",
        "wall_time_s",
        "tokens_per_second",
        "qps",
    ]

    rows = sorted(rows, key=lambda r: r["concurrency"])

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in keys})


def append_jsonl(jsonl_path, row):
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def dedupe_keep_last(rows):
    latest = {}
    for r in rows:
        latest[r["concurrency"]] = r
    return sorted(latest.values(), key=lambda r: r["concurrency"])


def choose_best(rows):
    max_tps = max(r["tokens_per_second"] for r in rows)
    candidates = [
        r for r in rows
        if r["tokens_per_second"] >= max_tps * 0.85
    ]
    return min(candidates, key=lambda r: r["avg_latency_s"])


def plot(rows, out_dir):
    rows = sorted(rows, key=lambda r: r["concurrency"])

    conc = [r["concurrency"] for r in rows]
    latency = [r["avg_latency_s"] for r in rows]
    tps = [r["tokens_per_second"] for r in rows]
    qps = [r["qps"] for r in rows]

    best = choose_best(rows)

    plt.figure()
    plt.plot(conc, latency, marker="o")
    plt.scatter([best["concurrency"]], [best["avg_latency_s"]], marker="x", s=120)
    plt.xlabel("Concurrency")
    plt.ylabel("Average latency (s)")
    plt.title("Concurrency vs Latency")
    plt.grid(True)
    plt.savefig(out_dir / "concurrency_latency.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(conc, tps, marker="o")
    plt.scatter([best["concurrency"]], [best["tokens_per_second"]], marker="x", s=120)
    plt.xlabel("Concurrency")
    plt.ylabel("Tokens per second")
    plt.title("Concurrency vs Throughput")
    plt.grid(True)
    plt.savefig(out_dir / "concurrency_throughput.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(latency, tps, marker="o")
    for r in rows:
        plt.annotate(str(r["concurrency"]), (r["avg_latency_s"], r["tokens_per_second"]))
    plt.scatter([best["avg_latency_s"]], [best["tokens_per_second"]], marker="x", s=120)
    plt.xlabel("Average latency (s)")
    plt.ylabel("Tokens per second")
    plt.title("Latency vs Throughput")
    plt.grid(True)
    plt.savefig(out_dir / "latency_throughput.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(conc, qps, marker="o")
    plt.xlabel("Concurrency")
    plt.ylabel("QPS")
    plt.title("Concurrency vs QPS")
    plt.grid(True)
    plt.savefig(out_dir / "concurrency_qps.png", dpi=200, bbox_inches="tight")
    plt.close()

    return best


def print_summary(rows, best):
    max_tps = max(r["tokens_per_second"] for r in rows)

    print("\n[Summary]")
    for r in rows:
        ratio = r["tokens_per_second"] / max_tps
        score = r["tokens_per_second"] / r["avg_latency_s"]
        print(
            f"c={r['concurrency']:>3}  "
            f"lat={r['avg_latency_s']:.2f}s  "
            f"tps={r['tokens_per_second']:.1f}  "
            f"qps={r['qps']:.2f}  "
            f"tps_ratio={ratio:.2%}  "
            f"score={score:.1f}"
        )

    print("\n[Recommended online concurrency]")
    print(best)

    batch_best = max(rows, key=lambda r: r["tokens_per_second"])
    print("\n[Best batch throughput concurrency]")
    print(batch_best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", default="scripts/benchmark_openai.py")
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--concurrency-list", default="1,2,4,8,16,32,64,128")
    ap.add_argument("--requests-multiplier", type=int, default=5)
    ap.add_argument("--min-requests", type=int, default=20)
    ap.add_argument("--out-dir", default="outputs/benchmark_sweep")
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--dedupe", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "results.csv"
    jsonl_path = out_dir / "results.jsonl"

    if args.append:
        rows = load_existing(csv_path)
    else:
        rows = []

    conc_list = [int(x) for x in args.concurrency_list.split(",")]

    for c in conc_list:
        req = max(args.min_requests, c * args.requests_multiplier)
        result = run_one(args.script, args.base_url, args.model, c, req)
        rows.append(result)
        append_jsonl(jsonl_path, result)

    if args.dedupe:
        rows = dedupe_keep_last(rows)

    save_csv(csv_path, rows)
    best = plot(rows, out_dir)
    print_summary(rows, best)

    print("\n[DONE]")
    print(f"CSV:   {csv_path}")
    print(f"JSONL:  {jsonl_path}")
    print(f"Plots: {out_dir}")


if __name__ == "__main__":
    main()