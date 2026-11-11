#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
import io

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

INPUT = RESULTS / "spatial_avg_RT_R1-4_T1-8_n1-100.csv"

OUT_L = (
    RESULTS /
    "spatial_avg_RT_R1-4_T1-8_n1-100_jackknife.csv"
)

OUT_V = (
    RESULTS /
    "laplace_Veff_R1-4_T1-8_n1-100_jackknife.csv"
)

EXPECTED_CFGS = list(range(1, 101))
EXPECTED_T = list(range(1, 9))
EXPECTED_R = list(range(1, 5))
EXPECTED_NSRC = 24 ** 3

def load_raw_log(filepath):
    data_lines = []
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            # Data lines have 6 items and start with an integer (the cfg number)
            if len(parts) == 6 and not parts[0].startswith(("#", "[", "R", "E", "A", "M", "T", "c")):
                try: 
                    # Verify first 4 are integers
                    int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]) 
                    data_lines.append(",".join(parts))
                except ValueError:
                    continue
    csv_data = "cfg,T,R,Nsrc,Re,Im\n" + "\n".join(data_lines)
    df = pd.read_csv(io.StringIO(csv_data)) # create an in-memory file from the strings, read the comma-separated text as .csv, transform it into a structured Dataframe
    df = df.drop_duplicates(subset=["cfg", "T", "R"]).reset_index(drop=True)
    return df 

def validate_input(df):
    required = {"cfg", "T", "R", "Nsrc", "Re", "Im"}

    missing_columns = required - set(df.columns) # check if there is a column missing

    if missing_columns:
        raise RuntimeError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    if df.duplicated(["cfg", "T", "R"]).any():
        duplicates = df[
            df.duplicated(
                ["cfg", "T", "R"],
                keep=False,
            )
        ]
        raise RuntimeError(
            "Duplicate (cfg,T,R) rows:\n"
            + duplicates.to_string(index=False)
        )

    cfgs = sorted(df["cfg"].unique().tolist())
    t_values = sorted(df["T"].unique().tolist())
    r_values = sorted(df["R"].unique().tolist())

    if cfgs != EXPECTED_CFGS:
        raise RuntimeError(
            f"Configuration IDs differ: {cfgs}"
        )

    if t_values != EXPECTED_T:
        raise RuntimeError(
            f"Unexpected T values: {t_values}"
        )

    if r_values != EXPECTED_R:
        raise RuntimeError(
            f"Unexpected R values: {r_values}"
        )

    if len(df) != 3200: # Ncfig * time slices
        raise RuntimeError(
            f"Expected 3200 rows, found {len(df)}"
        )

    bad_nsrc = df[df["Nsrc"] != EXPECTED_NSRC]

    if not bad_nsrc.empty:
        raise RuntimeError(
            "Unexpected Nsrc values:\n"
            + bad_nsrc.to_string(index=False)
        )

    expected_grid = {
        (T, R)
        for T in EXPECTED_T
        for R in EXPECTED_R
    }

    for cfg, group in df.groupby("cfg"):
        grid = set(zip(group["T"], group["R"]))

        if grid != expected_grid:
            raise RuntimeError(
                f"Grid mismatch for cfg={cfg}: "
                f"missing={sorted(expected_grid - grid)}, "
                f"extra={sorted(grid - expected_grid)}"
            )

    return cfgs


def jackknife_error(samples):
    samples = np.asarray(samples, dtype=float)
    n = len(samples)

    if n < 2:
        return np.nan

    center = samples.mean()

    return np.sqrt(
        (n - 1) / n
        * np.sum((samples - center) ** 2)
    )


def main():

    df = load_raw_log(INPUT)
    df = df.sort_values(["cfg", "T", "R"]).reset_index(drop=True)

    cfgs = validate_input(df)
    ncfg = len(cfgs)

    arrays_re = {}
    arrays_im = {}

    for T in EXPECTED_T:
        for R in EXPECTED_R:
            group = (
                df[
                    (df["T"] == T) &
                    (df["R"] == R)
                ]
                .sort_values("cfg")
            )

            if group["cfg"].tolist() != cfgs:
                raise RuntimeError(
                    f"Configuration mismatch at T={T}, R={R}"
                )
            # Select array of L_re
            arrays_re[(T, R)] = (
                group["Re"].to_numpy(dtype=float)
            )
            # Select array of L_im
            arrays_im[(T, R)] = (
                group["Im"].to_numpy(dtype=float)
            )

    rows_L = []

    for T in EXPECTED_T:
        for R in EXPECTED_R:
            re_values = arrays_re[(T, R)]
            im_values = arrays_im[(T, R)]

            sum_re = re_values.sum()
            sum_im = im_values.sum()

            # Compute an array of 100 sum_re, each missing one configuration, then average over (ncfg -1)
            jk_re = (
                sum_re - re_values
            ) / (ncfg - 1)

            jk_im = (
                sum_im - im_values
            ) / (ncfg - 1)

            # Jackknife error for L - correlator
            err_re = jackknife_error(jk_re)
            err_im = jackknife_error(jk_im)

            mean_re = re_values.mean()
            mean_im = im_values.mean()

            relative_error = (
                100.0 * err_re / abs(mean_re)
                if mean_re != 0.0
                else np.nan
            )

            snr = (
                abs(mean_re) / err_re
                if err_re > 0.0
                else np.inf
            )

            rows_L.append({
                "T": T,
                "R": R,
                "Ncfg": ncfg,
                "mean_Re": mean_re,
                "err_Re_jackknife": err_re,
                "relative_err_Re_percent": relative_error,
                "snr_Re": snr,
                "n_nonpositive_Re": int(
                    np.count_nonzero(re_values <= 0.0)
                ),
                "mean_Im": mean_im,
                "err_Im_jackknife": err_im,
            })

    rows_V = []
    # Using 2-step finite difference to handle for T odd 
    for T in EXPECTED_T[:-2]:
        for R in EXPECTED_R:
            values_T = arrays_re[(T, R)] # L at T,R
            values_Tp2 = arrays_re[(T + 2, R)] # L at T+2, R

            mean_T = values_T.mean()
            mean_Tp2 = values_Tp2.mean()

            jk_mean_T = (
                values_T.sum() - values_T
            ) / (ncfg - 1)

            jk_mean_Tp2 = (
                values_Tp2.sum() - values_Tp2
            ) / (ncfg - 1)

            valid_jk = (
                (jk_mean_T > 0.0) &
                (jk_mean_Tp2 > 0.0)
            )

            n_valid = int(valid_jk.sum())

            if mean_T <= 0.0 or mean_Tp2 <= 0.0:
                veff = np.nan
                err_veff = np.nan
                status = "nonpositive_central_mean"

            elif n_valid != ncfg:
                veff = 0.5 * np.log(mean_T / mean_Tp2)
                err_veff = np.nan
                status = "nonpositive_leave_one_out_mean"

            else: 
                # Compute the V_eff
                veff = 0.5 * np.log(mean_T / mean_Tp2)

                jk_veff = 0.5 * np.log(
                    jk_mean_T / jk_mean_Tp2
                )

                err_veff = jackknife_error(jk_veff)
                status = "ok"

            relative_error = (
                100.0 * err_veff / abs(veff)
                if (
                    np.isfinite(veff) and
                    np.isfinite(err_veff) and
                    veff != 0.0
                )
                else np.nan
            )

            rows_V.append({
                "T": T,
                "R": R,
                "Ncfg": ncfg,
                "mean_Re_T": mean_T,
                "mean_Re_Tp2": mean_Tp2,
                "Veff_log_ratio_of_means": veff,
                "err_Veff_jackknife": err_veff,
                "relative_err_Veff_percent": relative_error,
                "n_valid_jackknife": n_valid,
                "status": status,
            })

    out_L = pd.DataFrame(rows_L)
    out_V = pd.DataFrame(rows_V)

    out_L.to_csv(
        OUT_L,
        index=False,
    )

    out_V.to_csv(
        OUT_V,
        index=False,

    )

    print(f"Input rows: {len(df)}")
    print(f"Configurations: {ncfg}")
    print(f"L(R,T) rows: {len(out_L)}")
    print(f"Veff rows: {len(out_V)}")

    print()
    print("Veff status counts:")
    print(
        out_V["status"]
        .value_counts(dropna=False)
        .to_string()
    )

    invalid = out_V[out_V["status"] != "ok"]

    if not invalid.empty:
        print()
        print("Non-standard Veff rows:")
        print(
            invalid[[
                "T",
                "R",
                "mean_Re_T",
                "mean_Re_Tp2",
                "n_valid_jackknife",
                "status",
            ]].to_string(index=False)
        )

    print()
    print("Wrote:")
    print(OUT_L)
    print(OUT_V)


if __name__ == "__main__":
    main()
