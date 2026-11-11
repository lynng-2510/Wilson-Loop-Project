#!/usr/bin/env python3

import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

figdir = (RESULTS/"figures")
os.makedirs(figdir, exist_ok=True)

L_path = (RESULTS/"spatial_avg_RT_R1-4_T1-8_n1-100_jackknife.csv")
V_path = (RESULTS/"laplace_Veff_R1-4_T1-8_n1-100_jackknife.csv")
L_rows = read_csv(L_path)
V_rows = read_csv(V_path)

# Convert rows
L = []
for r in L_rows:
    L.append({
        "T": int(r["T"]),
        "R": int(r["R"]),
        "mean_Re": float(r["mean_Re"]),
        "err_Re": float(r["err_Re_jackknife"]),
        "mean_Im": float(r["mean_Im"]),
        "err_Im": float(r["err_Im_jackknife"]),
    })

V = []
for r in V_rows:
    V.append({
        "T": int(r["T"]),
        "R": int(r["R"]),
        "Veff": float(r["Veff_log_ratio_of_means"]),
        "err": float(r["err_Veff_jackknife"]),
    })


# 1) Re L(R,T) vs T
plt.figure(figsize=(7.0, 4.8))

for R in range(1, 5):
    xs = [r["T"] for r in L if r["R"] == R]
    ys = [r["mean_Re"] for r in L if r["R"] == R]
    es = [r["err_Re"] for r in L if r["R"] == R]

    plt.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=f"R/a = {R}")

plt.yscale("log")
plt.xlabel("T/a")
plt.ylabel("Re L(R,T)")
plt.title("Laplace correlator with jackknife errors")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
out = f"{figdir}/laplace_ReL_vs_T_jackknife.png"
plt.savefig(out, dpi=200)
plt.close()
print(out)


# 2) Veff(R,T) vs T for physical R > 0
plt.figure(figsize=(7.0, 4.8))

for R in range(1, 5):
    xs = [r["T"] for r in V if r["R"] == R]
    ys = [r["Veff"] for r in V if r["R"] == R]
    es = [r["err"] for r in V if r["R"] == R]

    plt.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=f"R/a = {R}")

plt.xlabel("T/a")
plt.ylabel("a Veff(R,T)")
plt.title("Laplace effective potential with jackknife errors")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
out = f"{figdir}/laplace_Veff_vs_T_jackknife.png"
plt.savefig(out, dpi=200)
plt.close()
print(out)


# 3) Veff(R,T) vs R for T = 1,2,3
plt.figure(figsize=(7.0, 4.8))

for T in range(1, 7):
    xs = [r["R"] for r in V if r["T"] == T and r["R"] > 0]
    ys = [r["Veff"] for r in V if r["T"] == T and r["R"] > 0]
    es = [r["err"] for r in V if r["T"] == T and r["R"] > 0]

    plt.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=f"T/a = {T}")

plt.xlabel("R/a")
plt.ylabel("a Veff(R,T)")
plt.title("Laplace effective potential versus separation")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
out = f"{figdir}/laplace_Veff_vs_R_jackknife.png"
plt.savefig(out, dpi=200)
plt.close()
print(out)


# 4) Im L(R,T) vs T sanity check
plt.figure(figsize=(7.0, 4.8))

for R in range(1, 5):
    xs = [r["T"] for r in L if r["R"] == R]
    ys = [r["mean_Im"] for r in L if r["R"] == R]
    es = [r["err_Im"] for r in L if r["R"] == R]

    plt.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=f"R/a = {R}")

plt.axhline(0.0, linestyle="--", linewidth=1)
plt.xlabel("T/a")
plt.ylabel("Im L(R,T)")
plt.title("Imaginary part sanity check")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
out = f"{figdir}/laplace_ImL_vs_T_jackknife.png"
plt.savefig(out, dpi=200)
plt.close()
print(out)
