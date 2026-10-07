"""
Generate the SIMULATED example dataset shipped with the app.

Design (hepatocellular carcinoma miRNA panel, illustrative only):
  Groups   : Normal (control), Cirrhosis, HCC — 6 biological samples each
  Targets  : miR-21-5p, miR-122-5p, miR-221-3p, miR-199a-3p
  References: RNU48, miR-16-5p (stable) and U6 (deliberately unstable)
  3 technical replicates per sample and assay.

True log2 fold changes vs Normal are written to data/example_truth.csv so the
tests can check that the app recovers them.
"""

import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(2026)
OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)

groups = ["Normal", "Cirrhosis", "HCC"]
base_ct = {"miR-21-5p": 24.0, "miR-122-5p": 22.0, "miR-221-3p": 27.0, "miR-199a-3p": 26.0,
           "RNU48": 25.0, "miR-16-5p": 23.0, "U6": 21.0}
# true log2 fold change (positive = up-regulated) relative to Normal
true_log2fc = {
    "miR-21-5p":   {"Normal": 0, "Cirrhosis": 0.7, "HCC": 2.0},
    "miR-122-5p":  {"Normal": 0, "Cirrhosis": -0.8, "HCC": -2.0},
    "miR-221-3p":  {"Normal": 0, "Cirrhosis": 0.3, "HCC": 1.5},
    "miR-199a-3p": {"Normal": 0, "Cirrhosis": -0.2, "HCC": -1.3},
}
bio_sd = {"RNU48": 0.08, "miR-16-5p": 0.08, "U6": 0.7}
u6_group_shift = {"Normal": 0.0, "Cirrhosis": 0.3, "HCC": -0.6}

rows = []
for g in groups:
    for i in range(1, 7):
        sample = f"{g[:3].upper()}-{i:02d}"
        loading = rng.normal(0, 0.6)  # RNA input difference, removed by normalisation
        for gene, ct0 in base_ct.items():
            if gene in true_log2fc:
                bio = ct0 - true_log2fc[gene][g] + rng.normal(0, 0.35)
            else:
                bio = ct0 + rng.normal(0, bio_sd[gene]) + (u6_group_shift[g] if gene == "U6" else 0)
            for rep in range(1, 4):
                ct = bio + loading + rng.normal(0, 0.12)
                rows.append({"Sample": sample, "Group": g, "Target": gene, "Replicate": rep, "Ct": round(ct, 2)})

df = pd.DataFrame(rows)
# Realistic problems for the QC step to catch
df.loc[(df.Sample == "NOR-03") & (df.Target == "miR-221-3p") & (df.Replicate == 2), "Ct"] += 1.6
df["Ct"] = df["Ct"].astype(object)
df.loc[(df.Sample == "HCC-04") & (df.Target == "miR-122-5p") & (df.Replicate == 3), "Ct"] = "Undetermined"

df.to_csv(OUT / "example_mirna_qpcr.csv", index=False)

truth = [{"Target": t, "Group": g, "True log2 FC": v} for t, d in true_log2fc.items() for g, v in d.items()]
pd.DataFrame(truth).to_csv(OUT / "example_truth.csv", index=False)

template = pd.DataFrame({
    "Sample": ["S1", "S1", "S1", "S1", "S2", "S2"],
    "Group": ["Control", "Control", "Control", "Control", "Treated", "Treated"],
    "Target": ["miR-21-5p", "miR-21-5p", "RNU48", "RNU48", "miR-21-5p", "RNU48"],
    "Ct": [24.10, 24.18, 25.02, 24.95, 22.31, 25.10],
})
template.to_csv(OUT / "template.csv", index=False)
print(f"Wrote {len(df)} rows")


# ---------------------------------------------------------------------------
# Example 2 (SIMULATED): plasma miRNA panel exported from a QuantStudio-style
# instrument, with no Group column (groups are guessed from sample names).
# Includes: exogenous spike-in cel-miR-39-3p, one hemolysed and one borderline
# sample (miR-451a), one extraction failure (all Cq late, including spike-in),
# NTC wells, and a low-abundance miRNA that is not detected in every sample.
# ---------------------------------------------------------------------------
rng2 = np.random.default_rng(39)
panel_base = {
    "miR-122-5p": 30.0, "miR-21-5p": 27.5, "miR-223-3p": 24.0, "miR-16-5p": 22.5,
    "miR-25-3p": 26.0, "miR-92a-3p": 25.0, "miR-126-3p": 26.5, "miR-150-5p": 27.0,
    "miR-191-5p": 26.8, "miR-24-3p": 25.5, "miR-23a-3p": 26.0, "miR-451a": 22.0,
    "let-7a-5p": 27.2, "miR-142-3p": 25.8, "miR-375-3p": 34.4, "miR-30d-5p": 28.0,
}
panel_effect_hcc = {"miR-122-5p": 2.5, "miR-21-5p": 1.0, "miR-223-3p": -1.0, "miR-375-3p": 1.5}
spike = "cel-miR-39-3p"

wells, well_no = [], 0
plate_rows = "ABCDEFGH"
def next_pos(n):
    k = (n - 1) % 96
    return f"{plate_rows[k // 12]}{k % 12 + 1}"

for grp in ("Healthy", "HCC"):
    for i in range(1, 9):
        name = f"{grp}_{i:02d}"
        plasma_input = rng2.normal(0, 0.4)      # affects endogenous miRNAs only
        extraction = rng2.normal(0, 0.25)        # affects everything, incl. spike-in
        if name == "HCC_05":
            extraction += 2.6                    # failed extraction
        values = {}
        for mir, base in panel_base.items():
            eff = panel_effect_hcc.get(mir, 0.0) if grp == "HCC" else 0.0
            v = base - eff + plasma_input + extraction + rng2.normal(0, 0.3)
            if mir in ("miR-451a",) and name == "Healthy_03":
                v -= 4.0                         # hemolysed: red-cell miR-451a released
            if mir in ("miR-16-5p",) and name == "Healthy_03":
                v -= 1.5
            if mir == "miR-451a" and name == "HCC_02":
                v -= 1.8                         # borderline
            values[mir] = v
        values[spike] = 23.0 + extraction + rng2.normal(0, 0.15)
        for mir, v in values.items():
            for rep in range(2):
                well_no += 1
                ct = v + rng2.normal(0, 0.12)
                ct_txt = "Undetermined" if ct > 36.0 else f"{ct:.3f}"
                wells.append([well_no, next_pos(well_no), "false", name, mir, "UNKNOWN", "FAM", "NFQ-MGB", ct_txt])
for mir in list(panel_base) + [spike]:
    well_no += 1
    wells.append([well_no, next_pos(well_no), "false", "NTC", mir, "NTC", "FAM", "NFQ-MGB", "Undetermined"])

header_lines = [
    "* Block Type = 96-Well Block (0.2mL)",
    "* Chemistry = TAQMAN",
    "* Experiment Name = Plasma miRNA panel (simulated example)",
    "* Instrument Type = QuantStudio(TM) 5 System",
    "* Passive Reference = ROX",
    "",
    "[Results]",
]
cols = ["Well", "Well Position", "Omit", "Sample Name", "Target Name", "Task", "Reporter", "Quencher", "CT"]
with open(OUT / "example_plasma_quantstudio.txt", "w") as fh:
    fh.write("\n".join(header_lines) + "\n")
    fh.write("\t".join(cols) + "\n")
    for w in wells:
        fh.write("\t".join(str(x) for x in w) + "\n")
print(f"Wrote plasma example with {len(wells)} wells")
