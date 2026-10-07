"""
Core calculations for the qPCR Fold-Change Analyzer.

Everything here is plain Python + numpy/pandas/scipy, with no Streamlit code,
so every function can be tested on its own.

Method summary
--------------
Relative quantities are computed with the efficiency-corrected, multi-reference
model used by qbase (Hellemans et al., Genome Biology 2007), working on the
log2 scale:

    log2 RQ(gene, sample) = (Ct_calibrator(gene) - Ct(gene, sample)) * log2(E_gene)
    log2 NF(sample)       = mean of log2 RQ over the reference genes
    log2 NRQ(gene,sample) = log2 RQ - log2 NF

where E_gene is the amplification factor (2.0 = 100 % efficiency) and the
calibrator is the mean Ct of that gene in the control group. When every
efficiency is 100 % this is exactly the Livak 2^-ΔΔCt method
(log2 NRQ = -ΔΔCt). Statistics are done on log2 NRQ (i.e. on -ΔΔCt), which is
the scale on which qPCR data are approximately normal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

# Standard column names used inside the app
SAMPLE, GROUP, TARGET, CT = "Sample", "Group", "Target", "Ct"

# Words that usually mean "no amplification"
UNDETERMINED_WORDS = {"undetermined", "undet", "na", "n/a", "nan", "no ct", "noct", "-", ""}

# Names that commonly indicate a reference / normaliser gene
REFERENCE_HINTS = (
    "u6", "rnu6", "rnu44", "rnu48", "rnu43", "snord", "snor", "5s", "18s",
    "gapdh", "actb", "b-actin", "beta-actin", "b2m", "hprt", "tbp", "ppia",
    "rplp0", "ywhaz", "mir-16", "mir16", "mir-103", "mir-191", "cel-mir-39",
    "cel-39", "spike",
)


# --------------------------------------------------------------------------
# 1. Loading and cleaning
# --------------------------------------------------------------------------

def guess_column(columns, candidates):
    """Return the first column whose lower-case name matches one of `candidates`."""
    lowered = {str(c).strip().lower(): c for c in columns}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    for cand in candidates:  # partial match as a fallback
        for low, original in lowered.items():
            if cand in low:
                return original
    return None


def guess_columns(df: pd.DataFrame) -> dict:
    """Best guess of which uploaded column is Sample / Group / Target / Ct."""
    cols = list(df.columns)
    return {
        SAMPLE: guess_column(cols, ["sample", "sample name", "sample_name", "id", "subject"]),
        GROUP: guess_column(cols, ["group", "condition", "treatment", "biological group", "class"]),
        TARGET: guess_column(cols, ["target", "target name", "gene", "mirna", "assay", "detector"]),
        CT: guess_column(cols, ["ct", "cq", "cт", "ct value", "cq value", "crt", "cp"]),
    }


def standardize(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """
    Rename user columns to Sample/Group/Target/Ct, clean text, and convert Ct
    to numbers. Non-numeric Ct values (e.g. 'Undetermined') become NaN.
    Adds a boolean column 'Undetermined'.
    """
    missing = [k for k in (SAMPLE, GROUP, TARGET, CT) if not mapping.get(k)]
    if missing:
        raise ValueError(f"Please choose a column for: {', '.join(missing)}")

    out = pd.DataFrame({
        SAMPLE: df[mapping[SAMPLE]].astype(str).str.strip(),
        GROUP: df[mapping[GROUP]].astype(str).str.strip(),
        TARGET: df[mapping[TARGET]].astype(str).str.strip(),
    })
    raw_ct = df[mapping[CT]]
    ct_text = raw_ct.astype(str).str.strip().str.replace(",", ".", regex=False)
    out[CT] = pd.to_numeric(ct_text, errors="coerce")
    out["Undetermined"] = out[CT].isna()

    # Drop completely empty rows (e.g. trailing blank lines in Excel)
    blank = out[[SAMPLE, TARGET]].isin(["", "nan", "None"]).any(axis=1)
    out = out.loc[~blank].reset_index(drop=True)

    # A sample must belong to exactly one group
    groups_per_sample = out.groupby(SAMPLE)[GROUP].nunique()
    conflicting = groups_per_sample[groups_per_sample > 1].index.tolist()
    if conflicting:
        raise ValueError(
            "These samples are assigned to more than one group: "
            + ", ".join(map(str, conflicting[:10]))
        )
    return out


def suggest_references(targets) -> list:
    """Targets whose names look like common reference genes."""
    found = []
    for t in targets:
        low = str(t).lower().replace("_", "-").replace(" ", "")
        if any(h in low for h in REFERENCE_HINTS):
            found.append(t)
    return found


# --------------------------------------------------------------------------
# 2. Technical replicate QC
# --------------------------------------------------------------------------

def _drop_furthest(values: np.ndarray) -> tuple[np.ndarray, float]:
    """Remove the single value furthest from the median; return (kept, removed)."""
    med = np.median(values)
    idx = int(np.argmax(np.abs(values - med)))
    return np.delete(values, idx), float(values[idx])


def replicate_qc(
    data: pd.DataFrame,
    sd_threshold: float = 0.5,
    late_ct: float = 35.0,
    remove_outlier: bool = False,
) -> pd.DataFrame:
    """
    Summarise technical replicates for every Sample x Target.

    If `remove_outlier` is True and a triplicate (or larger) has SD above the
    threshold, the replicate furthest from the median is removed once.
    Returns one row per Sample x Target with the mean Ct used downstream.
    """
    rows = []
    for (sample, group, target), sub in data.groupby([SAMPLE, GROUP, TARGET], sort=False):
        all_ct = sub[CT].to_numpy(dtype=float)
        valid = all_ct[~np.isnan(all_ct)]
        n_total, n_undet = len(all_ct), int(np.isnan(all_ct).sum())
        sd_before = float(np.std(valid, ddof=1)) if len(valid) > 1 else np.nan

        used = valid
        removed = None
        if remove_outlier and len(valid) >= 3 and sd_before > sd_threshold:
            used, removed = _drop_furthest(valid)

        mean_ct = float(np.mean(used)) if len(used) else np.nan
        sd_after = float(np.std(used, ddof=1)) if len(used) > 1 else np.nan

        flags = []
        if len(valid) == 0:
            flags.append("No amplification")
        if n_undet and len(valid):
            flags.append(f"{n_undet} undetermined")
        if not np.isnan(sd_after) and sd_after > sd_threshold:
            flags.append(f"High SD (>{sd_threshold:g})")
        if len(used) == 1 and n_total > 1:
            flags.append("Single usable replicate")
        if not np.isnan(mean_ct) and mean_ct > late_ct:
            flags.append(f"Late Ct (>{late_ct:g})")
        if removed is not None:
            flags.append(f"Outlier removed (Ct {removed:.2f})")

        rows.append({
            SAMPLE: sample, GROUP: group, TARGET: target,
            "Replicates": n_total, "Used": len(used),
            "Mean Ct": mean_ct,
            "SD (raw)": sd_before, "SD (used)": sd_after,
            "Flags": "; ".join(flags),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 3. Relative quantification
# --------------------------------------------------------------------------

def efficiency_to_factor(efficiency_percent: float) -> float:
    """100 % efficiency -> amplification factor 2.0."""
    return 1.0 + float(efficiency_percent) / 100.0


def global_mean_genes(qc: pd.DataFrame, max_ct: float = 35.0, exclude=()) -> list:
    """
    Assays used for global mean normalisation (Mestdagh et al., Genome Biology 2009):
    those detected (mean Ct present and <= max_ct) in every sample.
    Spike-ins and other technical controls should be passed in `exclude`.
    """
    wide = qc.pivot_table(index=SAMPLE, columns=TARGET, values="Mean Ct", aggfunc="first")
    wide = wide[[c for c in wide.columns if c not in set(exclude)]]
    detected = wide.notna() & (wide <= max_ct)
    return [g for g in wide.columns if detected[g].all()]


def relative_quantities(
    qc: pd.DataFrame,
    references: list,
    control_group: str,
    efficiencies: dict | None = None,
    report_targets: list | None = None,
) -> pd.DataFrame:
    """
    Calculate ΔCt, ΔΔCt, log2 fold change and fold change per sample and target.

    Parameters
    ----------
    qc : output of `replicate_qc` (needs Sample, Group, Target, Mean Ct)
    references : genes forming the normaliser: reference genes, a spike-in,
                 or every assay in the global mean
    control_group : the group used as calibrator (fold change = 1)
    efficiencies : {gene: efficiency %}; missing genes default to 100 %
    report_targets : genes to report. Defaults to every gene not in `references`;
                     with global mean normalisation the targets are part of the normaliser.
    """
    if not references:
        raise ValueError("Select at least one reference gene.")
    efficiencies = efficiencies or {}

    wide = qc.pivot_table(index=[SAMPLE, GROUP], columns=TARGET, values="Mean Ct", aggfunc="first")
    missing_refs = [r for r in references if r not in wide.columns]
    if missing_refs:
        raise ValueError(f"Reference gene(s) not found in data: {', '.join(missing_refs)}")
    groups = wide.index.get_level_values(GROUP)
    if control_group not in set(groups):
        raise ValueError(f"Control group '{control_group}' not found.")

    is_control = groups == control_group
    factor = {g: efficiency_to_factor(efficiencies.get(g, 100.0)) for g in wide.columns}

    # Calibrator Ct = mean Ct of each gene in the control group
    calibrator = wide.loc[is_control].mean(axis=0, skipna=True)

    log2_rq = pd.DataFrame(index=wide.index)
    for g in wide.columns:
        log2_rq[g] = (calibrator[g] - wide[g]) * math.log2(factor[g])

    # Normalisation factor: mean log2 RQ of references (= geometric mean of RQ).
    # A sample missing any reference gets NaN (not silently normalised to fewer genes).
    log2_nf = log2_rq[references].mean(axis=1, skipna=False)
    ref_ct_mean = wide[references].mean(axis=1, skipna=False)

    if report_targets is not None:
        targets = [g for g in report_targets if g in wide.columns]
    else:
        targets = [g for g in wide.columns if g not in references]
    records = []
    for g in targets:
        log2_nrq = log2_rq[g] - log2_nf
        for (sample, group), val in log2_nrq.items():
            ct_t = wide.loc[(sample, group), g]
            ref_ct = ref_ct_mean.loc[(sample, group)]
            records.append({
                SAMPLE: sample, GROUP: group, TARGET: g,
                "Target Ct": ct_t,
                "Reference Ct (mean)": ref_ct,
                "ΔCt": ct_t - ref_ct,
                "ΔΔCt": -val,
                "log2 FC": val,
                "Fold change": 2.0 ** val if pd.notna(val) else np.nan,
            })
    out = pd.DataFrame(records)
    if out.empty:
        raise ValueError("No target genes left after removing the reference genes.")
    return out


# --------------------------------------------------------------------------
# 4. Group summaries and statistics
# --------------------------------------------------------------------------

def group_summary(results: pd.DataFrame, group_order: list | None = None) -> pd.DataFrame:
    """
    Per Target x Group: n, mean log2 FC, SD, SEM, 95 % CI and the back-transformed
    (geometric mean) fold change with its 95 % CI. The arithmetic mean of
    2^-ΔΔCt ± SEM is included as well because many labs still report it.
    """
    rows = []
    for (target, group), sub in results.groupby([TARGET, GROUP], sort=False):
        x = sub["log2 FC"].dropna().to_numpy()
        fc = sub["Fold change"].dropna().to_numpy()
        n = len(x)
        mean = float(np.mean(x)) if n else np.nan
        sd = float(np.std(x, ddof=1)) if n > 1 else np.nan
        sem = sd / math.sqrt(n) if n > 1 else np.nan
        tcrit = stats.t.ppf(0.975, n - 1) if n > 1 else np.nan
        lo, hi = (mean - tcrit * sem, mean + tcrit * sem) if n > 1 else (np.nan, np.nan)
        rows.append({
            TARGET: target, GROUP: group, "n": n,
            "Mean log2 FC": mean, "SD log2 FC": sd, "SEM log2 FC": sem,
            "Fold change (geometric mean)": 2 ** mean if n else np.nan,
            "95% CI low": 2 ** lo if n > 1 else np.nan,
            "95% CI high": 2 ** hi if n > 1 else np.nan,
            "Fold change (arithmetic mean)": float(np.mean(fc)) if len(fc) else np.nan,
            "SEM (arithmetic)": float(np.std(fc, ddof=1) / math.sqrt(len(fc))) if len(fc) > 1 else np.nan,
        })
    out = pd.DataFrame(rows)
    if group_order:
        out[GROUP] = pd.Categorical(out[GROUP], categories=group_order, ordered=True)
        out = out.sort_values([TARGET, GROUP]).reset_index(drop=True)
        out[GROUP] = out[GROUP].astype(str)
    return out


def p_adjust(pvalues, method: str = "holm") -> np.ndarray:
    """
    Multiple-testing correction, matching R's p.adjust for 'holm', 'BH' and 'none'.
    NaN p-values are ignored and stay NaN.
    """
    p = np.asarray(pvalues, dtype=float)
    out = np.full_like(p, np.nan)
    mask = ~np.isnan(p)
    q = p[mask]
    m = len(q)
    if m == 0:
        return out
    method = method.lower()
    if method == "none":
        adj = q.copy()
    elif method == "holm":
        order = np.argsort(q)
        ranked = q[order] * (m - np.arange(m))
        ranked = np.minimum(1.0, np.maximum.accumulate(ranked))
        adj = np.empty(m)
        adj[order] = ranked
    elif method in ("bh", "fdr"):
        order = np.argsort(q)[::-1]
        ranked = q[order] * m / (m - np.arange(m))
        ranked = np.minimum(1.0, np.minimum.accumulate(ranked))
        adj = np.empty(m)
        adj[order] = ranked
    else:
        raise ValueError(f"Unknown correction method: {method}")
    out[mask] = adj
    return out


TESTS = {
    "welch": "Welch's t-test (unequal variances)",
    "student": "Student's t-test (equal variances)",
    "dunnett": "Dunnett's test (each group vs control, adjusted within each target)",
    "mannwhitney": "Mann-Whitney U test (non-parametric)",
}


def _dunnett_pvalues(groups: list, control: np.ndarray) -> np.ndarray:
    """Dunnett many-to-one p-values with a fixed seed (SciPy's algorithm is quasi-random)."""
    try:
        res = stats.dunnett(*groups, control=control, rng=np.random.default_rng(12345))
    except TypeError:  # SciPy < 1.15 uses random_state
        res = stats.dunnett(*groups, control=control, random_state=np.random.default_rng(12345))
    return np.asarray(res.pvalue, dtype=float)
CORRECTIONS = {
    "holm": "Holm",
    "bh": "Benjamini-Hochberg (FDR)",
    "none": "None",
}


def significance_stars(p: float) -> str:
    if p is None or np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def compare_groups(
    results: pd.DataFrame,
    control_group: str,
    test: str = "welch",
    correction: str = "holm",
) -> pd.DataFrame:
    """
    Compare every non-control group with the control for each target, on the
    log2 FC (-ΔΔCt) scale. P-values are corrected across all comparisons in
    the table. If there are three or more groups an omnibus test
    (one-way ANOVA, or Kruskal-Wallis for the non-parametric option) is added.
    """
    rows = []
    groups_all = list(dict.fromkeys(results[GROUP]))
    for target, sub in results.groupby(TARGET, sort=False):
        ctrl = sub.loc[sub[GROUP] == control_group, "log2 FC"].dropna().to_numpy()

        omnibus_p = np.nan
        present = [g for g in groups_all if (sub[GROUP] == g).any()]
        samples = [sub.loc[sub[GROUP] == g, "log2 FC"].dropna().to_numpy() for g in present]
        samples = [s for s in samples if len(s) >= 2]
        if len(samples) >= 3:
            if test == "mannwhitney":
                omnibus_p = float(stats.kruskal(*samples).pvalue)
            else:
                omnibus_p = float(stats.f_oneway(*samples).pvalue)

        dunnett_p = {}
        if test == "dunnett" and len(ctrl) >= 2:
            testable = [g for g in present if g != control_group
                        and sub.loc[sub[GROUP] == g, "log2 FC"].dropna().size >= 2]
            if testable:
                pv = _dunnett_pvalues(
                    [sub.loc[sub[GROUP] == g, "log2 FC"].dropna().to_numpy() for g in testable], ctrl)
                dunnett_p = dict(zip(testable, pv))

        for g in present:
            if g == control_group:
                continue
            x = sub.loc[sub[GROUP] == g, "log2 FC"].dropna().to_numpy()
            p = np.nan
            if test == "dunnett":
                p = float(dunnett_p.get(g, np.nan))
            elif len(x) >= 2 and len(ctrl) >= 2:
                if test == "welch":
                    p = float(stats.ttest_ind(x, ctrl, equal_var=False).pvalue)
                elif test == "student":
                    p = float(stats.ttest_ind(x, ctrl, equal_var=True).pvalue)
                elif test == "mannwhitney":
                    p = float(stats.mannwhitneyu(x, ctrl, alternative="two-sided").pvalue)
                else:
                    raise ValueError(f"Unknown test: {test}")
            diff = float(np.mean(x) - np.mean(ctrl)) if len(x) and len(ctrl) else np.nan
            rows.append({
                TARGET: target,
                "Comparison": f"{g} vs {control_group}",
                "Group": g,
                "n (group)": len(x), "n (control)": len(ctrl),
                "Δ mean log2 FC": diff,
                "Fold change": 2 ** diff if not np.isnan(diff) else np.nan,
                "p-value": p,
                "Omnibus p (all groups)": omnibus_p,
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["Adjusted p"] = p_adjust(out["p-value"].to_numpy(), correction)
    out["Significance"] = [significance_stars(p) for p in out["Adjusted p"]]
    return out


# --------------------------------------------------------------------------
# 5. Reference gene stability (geNorm)
# --------------------------------------------------------------------------

def genorm_m(log2_quantities: pd.DataFrame) -> pd.Series:
    """
    geNorm stability value M (Vandesompele et al., Genome Biology 2002).

    `log2_quantities` has one column per gene and one row per sample, holding
    log2 relative quantities (e.g. -Ct * log2(E)). For each gene j,
    M_j = mean over k != j of SD( log2 Q_j - log2 Q_k ). Lower M = more stable.
    Only samples with values for every gene are used.
    """
    data = log2_quantities.dropna(axis=0, how="any")
    genes = list(data.columns)
    if len(genes) < 2:
        raise ValueError("geNorm needs at least two candidate genes.")
    if len(data) < 2:
        raise ValueError("geNorm needs at least two samples with all genes measured.")
    m = {}
    for j in genes:
        vs = [np.std(data[j] - data[k], ddof=1) for k in genes if k != j]
        m[j] = float(np.mean(vs))
    return pd.Series(m, name="M").sort_values()


def genorm_ranking(
    qc: pd.DataFrame,
    candidates: list,
    efficiencies: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full geNorm procedure: compute M, remove the least stable gene, repeat.

    Returns
    -------
    final : one row per candidate gene with its M (from the full set),
            the elimination order and a stability verdict.
    steps : M values at every elimination step.
    """
    efficiencies = efficiencies or {}
    wide = qc.pivot_table(index=SAMPLE, columns=TARGET, values="Mean Ct", aggfunc="first")
    wide = wide[[c for c in candidates if c in wide.columns]]
    log2q = pd.DataFrame({
        g: -wide[g] * math.log2(efficiency_to_factor(efficiencies.get(g, 100.0)))
        for g in wide.columns
    })

    full_m = genorm_m(log2q)
    remaining = list(log2q.columns)
    steps, eliminated = [], []
    while len(remaining) >= 2:
        m = genorm_m(log2q[remaining])
        steps.append({"Genes in set": len(remaining), **{f"M {g}": m[g] for g in remaining}})
        if len(remaining) == 2:
            break
        worst = m.idxmax()
        eliminated.append(worst)
        remaining.remove(worst)

    # Most stable first: final pair (tied), then genes in reverse elimination order
    ranking = remaining + list(reversed(eliminated))
    final = pd.DataFrame({
        "Gene": ranking,
        "Stability rank": range(1, len(ranking) + 1),
        "geNorm M (all candidates)": [full_m[g] for g in ranking],
    })
    final["Verdict"] = np.where(
        final["geNorm M (all candidates)"] <= 0.5, "Stable (M ≤ 0.5)",
        np.where(final["geNorm M (all candidates)"] <= 1.0, "Acceptable (M ≤ 1.0)", "Unstable (M > 1.0)"),
    )
    return final, pd.DataFrame(steps)


def reference_ct_spread(qc: pd.DataFrame, candidates: list) -> pd.DataFrame:
    """Simple descriptive check: SD and range of raw Ct across samples and groups."""
    rows = []
    for g in candidates:
        sub = qc.loc[qc[TARGET] == g]
        ct = sub["Mean Ct"].dropna()
        by_group = sub.groupby(GROUP)["Mean Ct"].mean()
        rows.append({
            "Gene": g,
            "Mean Ct": ct.mean(),
            "SD across samples": ct.std(ddof=1),
            "Range": ct.max() - ct.min(),
            "Max difference between group means": by_group.max() - by_group.min(),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 6. Primer efficiency from a standard curve
# --------------------------------------------------------------------------

@dataclass
class StandardCurve:
    slope: float
    intercept: float
    r_squared: float
    amplification_factor: float
    efficiency_percent: float
    n_points: int
    notes: list = field(default_factory=list)


def standard_curve(quantity, ct, log_input: bool = False) -> StandardCurve:
    """
    Fit Ct = slope * log10(quantity) + intercept.
    E = 10^(-1/slope); efficiency % = (E - 1) * 100.
    Set `log_input=True` if `quantity` is already log10-transformed.
    """
    q = np.asarray(quantity, dtype=float)
    c = np.asarray(ct, dtype=float)
    ok = ~(np.isnan(q) | np.isnan(c))
    q, c = q[ok], c[ok]
    if not log_input:
        if np.any(q <= 0):
            raise ValueError("Quantities must be positive (or tick 'already log10').")
        q = np.log10(q)
    if len(np.unique(q)) < 3:
        raise ValueError("Use at least three dilution levels.")
    fit = stats.linregress(q, c)
    e = 10 ** (-1.0 / fit.slope)
    eff = (e - 1) * 100
    notes = []
    if not 90 <= eff <= 110:
        notes.append("Efficiency outside the usual 90–110 % range.")
    if fit.rvalue ** 2 < 0.98:
        notes.append("R² below 0.98: check pipetting or remove the most dilute point.")
    return StandardCurve(fit.slope, fit.intercept, fit.rvalue ** 2, e, eff, len(q), notes)


# --------------------------------------------------------------------------
# 7. Sample quality for circulating miRNA: hemolysis and spike-in
# --------------------------------------------------------------------------

def find_assay(targets, patterns) -> str | None:
    """First target whose simplified name contains any of `patterns` (e.g. '451')."""
    for t in targets:
        low = str(t).lower().replace("_", "-").replace(" ", "")
        if any(p in low for p in patterns):
            return t
    return None


def hemolysis_check(
    qc: pd.DataFrame,
    mir23a: str,
    mir451a: str,
    high: float = 7.0,
    borderline: float = 5.0,
) -> pd.DataFrame:
    """
    Hemolysis indicator for serum/plasma samples (Blondal et al., Methods 2013):
    ΔCq = Cq(miR-23a) - Cq(miR-451a). Red blood cells release miR-451a, so
    hemolysis lowers its Cq and raises ΔCq. ΔCq above ~7 indicates a high risk
    of hemolysis; values between ~5 and 7 are borderline.
    """
    wide = qc.pivot_table(index=[SAMPLE, GROUP], columns=TARGET, values="Mean Ct", aggfunc="first")
    for name in (mir23a, mir451a):
        if name not in wide.columns:
            raise ValueError(f"{name} not found in the data.")
    d = wide[mir23a] - wide[mir451a]
    out = d.rename("ΔCq (miR-23a − miR-451a)").reset_index()
    out[f"Cq {mir23a}"] = wide[mir23a].to_numpy()
    out[f"Cq {mir451a}"] = wide[mir451a].to_numpy()
    out["Hemolysis risk"] = np.select(
        [out["ΔCq (miR-23a − miR-451a)"].isna(),
         out["ΔCq (miR-23a − miR-451a)"] >= high,
         out["ΔCq (miR-23a − miR-451a)"] >= borderline],
        ["Not assessable", "High", "Borderline"], default="Low")
    return out


def spike_in_qc(qc: pd.DataFrame, spike: str, tolerance: float = 1.0) -> pd.DataFrame:
    """
    Exogenous spike-in (e.g. cel-miR-39) added at a fixed amount before extraction.
    Its Cq should be nearly identical in every sample; a sample whose Cq differs
    from the median by more than `tolerance` cycles suggests poor extraction or
    RT inhibition.
    """
    sub = qc.loc[qc[TARGET] == spike, [SAMPLE, GROUP, "Mean Ct"]].rename(columns={"Mean Ct": f"Cq {spike}"})
    if sub.empty:
        raise ValueError(f"{spike} not found in the data.")
    med = sub[f"Cq {spike}"].median()
    sub["Difference from median"] = sub[f"Cq {spike}"] - med
    sub["Status"] = np.where(sub[f"Cq {spike}"].isna(), "Not detected",
                             np.where(sub["Difference from median"].abs() > tolerance, "Check sample", "OK"))
    return sub.reset_index(drop=True)


# --------------------------------------------------------------------------
# 8. Instrument export files (QuantStudio, Bio-Rad CFX, generic)
# --------------------------------------------------------------------------

_CT_NAMES = {"ct", "cq", "cт", "crt", "cp", "c t", "cq (δrn)"}
_SAMPLE_NAMES = {"sample", "sample name", "samplename", "sample id", "name"}
_TARGET_NAMES = {"target", "target name", "targetname", "detector", "detector name", "gene", "assay", "mirna"}


def _norm_cell(x) -> str:
    return str(x).strip().lower() if x is not None and not (isinstance(x, float) and np.isnan(x)) else ""


def rows_from_text(text: str) -> list:
    """Split a delimited text export into rows, tolerating metadata lines of any width."""
    import csv
    lines = [ln for ln in text.splitlines()]
    body = [ln for ln in lines if ln and not ln.startswith(("*", "#"))] or lines
    sample = "\n".join(body[:200])
    counts = {d: sample.count(d) for d in ("\t", ";", ",")}
    delim = max(counts, key=counts.get) if max(counts.values()) else ","
    return [row for row in csv.reader(lines, delimiter=delim)]


def find_table(rows: list, max_scan: int = 300) -> pd.DataFrame:
    """
    Locate the results table inside an instrument export that may start with
    metadata lines. The header row is the first row containing a Ct/Cq column
    plus a sample or target column.
    """
    for i, row in enumerate(rows[:max_scan]):
        cells = [_norm_cell(c) for c in row]
        has_ct = any(c in _CT_NAMES for c in cells)
        has_id = any(c in _SAMPLE_NAMES for c in cells) or any(c in _TARGET_NAMES for c in cells)
        if has_ct and has_id:
            header = [str(c).strip() for c in row]
            width = len(header)
            body = [list(r[:width]) + [None] * (width - len(r)) for r in rows[i + 1:]]
            df = pd.DataFrame(body, columns=header)
            df = df.loc[:, [c for c in df.columns if c]]  # drop unnamed columns
            df = df.replace({"": np.nan}).dropna(how="all")
            return df.reset_index(drop=True)
    raise ValueError("Could not find a results table with Sample/Target and Ct/Cq columns.")


def clean_instrument_table(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Detect the instrument format and remove non-sample wells (NTC, standards,
    positive controls). Returns the cleaned table and a short report.
    """
    cols = {c.strip().lower(): c for c in df.columns}
    fmt = "Generic table"
    if "well position" in cols or "task" in cols:
        fmt = "Applied Biosystems (QuantStudio / 7500 / StepOne)"
    elif "fluor" in cols and "content" in cols:
        fmt = "Bio-Rad CFX"
    before = len(df)
    out = df.copy()
    if "task" in cols:
        task = out[cols["task"]].astype(str).str.strip().str.upper()
        out = out[task.isin(["UNKNOWN", "UNKN", "NAN", ""])]
    if "content" in cols:
        content = out[cols["content"]].astype(str).str.strip().str.lower()
        out = out[~content.str.startswith(("ntc", "std", "neg", "pos", "nrt", "ntr"))]
    sample_col = guess_column(list(out.columns), ["sample", "sample name", "samplename", "name"])
    if sample_col is not None:
        s = out[sample_col].astype(str).str.strip()
        out = out[~s.str.lower().isin(["", "nan", "none", "ntc"])]
    report = {"format": fmt, "wells_in_file": before, "control_wells_removed": before - len(out)}
    return out.reset_index(drop=True), report


def guess_group(sample: str) -> str:
    """Guess a biological group from a sample name: 'NOR-03' -> 'NOR', 'Control 2' -> 'Control'."""
    import re
    s = str(sample).strip()
    g = re.sub(r"[\s_\-.]*\(?[A-Za-z]?\d+\)?$", "", s)
    g = re.sub(r"[\s_\-.]+$", "", g)
    return g if g else s


# --------------------------------------------------------------------------
# 9. Planning the next experiment: sample size
# --------------------------------------------------------------------------

def t_test_power(n: int, effect_log2: float, sd: float, alpha: float = 0.05) -> float:
    """Power of a two-sided, two-sample t-test with n per group."""
    if n < 2 or sd <= 0:
        return np.nan
    d = abs(effect_log2) / sd
    df = 2 * n - 2
    ncp = d * math.sqrt(n / 2)
    tcrit = stats.t.ppf(1 - alpha / 2, df)
    return float(stats.nct.sf(tcrit, df, ncp) + stats.nct.cdf(-tcrit, df, ncp))


def n_per_group(effect_log2: float, sd: float, alpha: float = 0.05, power: float = 0.8, max_n: int = 500):
    """Smallest n per group reaching the requested power (None if above max_n)."""
    if sd <= 0 or effect_log2 == 0 or np.isnan(sd):
        return None
    for n in range(2, max_n + 1):
        if t_test_power(n, effect_log2, sd, alpha) >= power:
            return n
    return None


def sample_size_table(results: pd.DataFrame, fold_changes=(1.5, 2.0), alpha=0.05, power=0.8) -> pd.DataFrame:
    """
    Per target: pooled within-group SD of log2 FC (from the current data) and the
    biological replicates per group needed to detect each fold change.
    """
    rows = []
    for target, sub in results.groupby(TARGET, sort=False):
        ss, dof = 0.0, 0
        for _, g in sub.groupby(GROUP):
            x = g["log2 FC"].dropna().to_numpy()
            if len(x) >= 2:
                ss += float(((x - x.mean()) ** 2).sum())
                dof += len(x) - 1
        sd = math.sqrt(ss / dof) if dof else np.nan
        row = {TARGET: target, "Pooled SD (log2 FC)": sd}
        for fc in fold_changes:
            n = n_per_group(math.log2(fc), sd, alpha, power) if not np.isnan(sd) else None
            row[f"n per group for {fc:g}-fold"] = n if n is not None else "> 500"
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 10. Written outputs: methods and results paragraphs
# --------------------------------------------------------------------------

NORMALISATION = {
    "reference": "Reference genes",
    "spike_in": "Exogenous spike-in",
    "global_mean": "Global mean (miRNA panels)",
}


def methods_text(
    references: list,
    control_group: str,
    efficiencies: dict,
    test: str,
    correction: str,
    sd_threshold: float,
    remove_outlier: bool,
    n_groups: int,
    normalisation: str = "reference",
    global_max_ct: float = 35.0,
    hemolysis_assessed: bool = False,
) -> str:
    refs = ", ".join(references)
    all_100 = all(abs(float(v) - 100.0) < 1e-9 for v in efficiencies.values()) if efficiencies else True
    eff_txt = (
        "assuming 100 % amplification efficiency for all assays (2^-ΔΔCt method; Livak and Schmittgen, 2001)"
        if all_100 else
        "using assay-specific amplification efficiencies (efficiency-corrected model of Hellemans et al., 2007)"
    )
    if normalisation == "global_mean":
        norm_txt = (f"the global mean of all {len(references)} assays detected (Ct ≤ {global_max_ct:g}) in every sample "
                    "(Mestdagh et al., 2009)")
    elif normalisation == "spike_in":
        norm_txt = f"the exogenous spike-in control ({refs})"
    else:
        norm_txt = (f"the geometric mean of the reference genes ({refs})" if len(references) > 1
                    else f"the reference gene ({refs})")
    outlier_txt = (
        f" When the SD of a technical replicate set exceeded {sd_threshold:g} cycles, the replicate furthest from the median was excluded."
        if remove_outlier else
        f" Technical replicate sets with SD above {sd_threshold:g} cycles were flagged."
    )
    hemo_txt = (" Sample hemolysis was assessed using the miR-23a/miR-451a ΔCq ratio (Blondal et al., 2013)."
                if hemolysis_assessed else "")
    omnibus = ""
    if n_groups >= 3:
        omnibus = (" An omnibus Kruskal-Wallis test" if test == "mannwhitney" else " An omnibus one-way ANOVA") + \
                  " across all groups is also reported."
    corr_txt = "" if correction == "none" else f", with {CORRECTIONS[correction]} correction for multiple comparisons"
    return (
        "Relative expression was calculated from the mean Ct of technical replicates."
        f"{outlier_txt}{hemo_txt} Expression of each target was normalised to {norm_txt} and expressed "
        f"relative to the {control_group} group, {eff_txt}. Statistical comparisons were performed on "
        f"log2-transformed relative quantities (−ΔΔCt) using {TESTS[test]}{corr_txt}.{omnibus} "
        "Fold changes are reported as geometric means with 95 % confidence intervals. "
        "P < 0.05 was considered significant."
    )


def _fmt_p(p: float) -> str:
    if p is None or np.isnan(p):
        return "p not available"
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def results_text(summary: pd.DataFrame, comparisons: pd.DataFrame, control_group: str,
                 adjusted: bool = True) -> str:
    """Draft results paragraph from the statistics table (to be edited by the author)."""
    if comparisons is None or comparisons.empty:
        return "Not enough biological replicates for statistical comparison."
    label = "adjusted " if adjusted else ""
    sentences, ns = [], []
    for _, c in comparisons.iterrows():
        s = summary[(summary[TARGET] == c[TARGET]) & (summary[GROUP] == c["Group"])]
        if s.empty or np.isnan(c["Adjusted p"]):
            continue
        s = s.iloc[0]
        fc = s["Fold change (geometric mean)"]
        lo, hi = s["95% CI low"], s["95% CI high"]
        ci = f", 95% CI {lo:.2f}–{hi:.2f}" if not (np.isnan(lo) or np.isnan(hi)) else ""
        if c["Adjusted p"] < 0.05:
            direction = "up-regulated" if fc > 1 else "down-regulated"
            size = f"{fc:.2f}-fold" if fc > 1 else f"{1 / fc:.2f}-fold (fold change {fc:.2f})"
            sentences.append(
                f"{c[TARGET]} was significantly {direction} in {c['Group']} compared with {control_group} "
                f"({size}{ci}; {label}{_fmt_p(c['Adjusted p'])}).")
        else:
            ns.append(f"{c[TARGET]} in {c['Group']} (fold change {fc:.2f}; {label}{_fmt_p(c['Adjusted p'])})")
    text = " ".join(sentences)
    if ns:
        text += (" " if text else "") + "No significant difference from " + control_group + " was found for " + \
                "; ".join(ns) + "."
    return text.strip()


REFERENCES_TEXT = [
    "Livak KJ, Schmittgen TD. Analysis of relative gene expression data using real-time quantitative PCR and the 2^-ΔΔCT method. Methods. 2001;25(4):402–408.",
    "Hellemans J, Mortier G, De Paepe A, Speleman F, Vandesompele J. qBase relative quantification framework and software for management and automated analysis of real-time quantitative PCR data. Genome Biol. 2007;8(2):R19.",
    "Vandesompele J, De Preter K, Pattyn F, et al. Accurate normalization of real-time quantitative RT-PCR data by geometric averaging of multiple internal control genes. Genome Biol. 2002;3(7):research0034.",
]
EXTRA_REFERENCES = {
    "global_mean": "Mestdagh P, Van Vlierberghe P, De Weer A, et al. A novel and universal method for microRNA RT-qPCR data normalization. Genome Biol. 2009;10(6):R64.",
    "hemolysis": "Blondal T, Jensby Nielsen S, Baker A, et al. Assessing sample and miRNA profile quality in serum and plasma or other biofluids. Methods. 2013;59(1):S1–S6.",
    "dunnett": "Dunnett CW. A multiple comparison procedure for comparing several treatments with a control. J Am Stat Assoc. 1955;50(272):1096–1121.",
}


def references_for(normalisation: str, test: str, hemolysis_assessed: bool) -> list:
    refs = list(REFERENCES_TEXT)
    if normalisation == "global_mean":
        refs.append(EXTRA_REFERENCES["global_mean"])
    if hemolysis_assessed:
        refs.append(EXTRA_REFERENCES["hemolysis"])
    if test == "dunnett":
        refs.append(EXTRA_REFERENCES["dunnett"])
    return refs
