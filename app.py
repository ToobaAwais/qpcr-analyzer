"""
qPCR Fold-Change Analyzer — Streamlit app, built for miRNA research.

Upload Ct values (or an instrument export) and get replicate QC, sample-quality
checks for circulating miRNA (hemolysis, spike-in), reference-gene stability,
three normalisation strategies, statistics, a publication-ready figure,
draft methods and results paragraphs, a sample-size planner and an Excel report.

Run locally:  streamlit run app.py
"""

import io
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

import core
import plots

# ---------------------------------------------------------------------------
# Your details: edit these lines before you deploy
# ---------------------------------------------------------------------------
APP_NAME = "qPCR Fold-Change Analyzer"
AUTHOR = "Tooba Mujtaba"
CONTACT_URL = "https://www.linkedin.com/company/novagenbioinformatics-company/"
CONTACT_LABEL = "NovaGen Bioinformatics on LinkedIn"
GITHUB_URL = "https://github.com/ToobaAwais/qpcr-analyzer"

EXAMPLES = {
    "Example: tissue miRNA, 3 groups": "data/example_mirna_qpcr.csv",
    "Example: plasma miRNA panel (QuantStudio export)": "data/example_plasma_quantstudio.txt",
}
UPLOAD = "Upload my file"
TEMPLATE_FILE = "data/template.csv"
ASSIGN = "(assign groups in the app)"
CONTROL_HINTS = ("control", "normal", "ctrl", "untreated", "vehicle", "veh", "nc", "healthy", "baseline", "wt", "mock",
                 "sham", "dmso", "pbs", "naive", "con")
SPIKE_HINTS = ("cel-mir", "cel-39", "cel-54", "cel-238", "spike", "unisp", "ath-mir")
SMALL_RNA_CONTROL_HINTS = ("u6", "rnu", "snord", "snor", "5s", "18s")

st.set_page_config(page_title=APP_NAME, page_icon="🧬", layout="wide")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_table(raw_bytes: bytes, name: str):
    """Read CSV/TXT/TSV or Excel, find the results table, remove NTC/standard wells."""
    name = name.lower()
    if name.endswith((".xlsx", ".xlsm")):
        xls = pd.ExcelFile(io.BytesIO(raw_bytes))
        sheets = xls.sheet_names
        default = next((i for i, s in enumerate(sheets) if "result" in s.lower()), 0)
        sheet = st.selectbox("Sheet", sheets, index=default) if len(sheets) > 1 else sheets[0]
        cells = pd.read_excel(xls, sheet_name=sheet, header=None, dtype=object)
        rows = cells.where(cells.notna(), "").values.tolist()
    else:
        rows = core.rows_from_text(raw_bytes.decode("utf-8-sig", errors="replace"))
    table = core.find_table(rows)
    table.columns = [str(c).strip() for c in table.columns]
    return core.clean_instrument_table(table)


def guess_control(groups):
    for g in groups:
        if str(g).strip().lower() in CONTROL_HINTS:
            return g
    for g in groups:  # partial matches only for longer, unambiguous words
        if any(h in str(g).lower() for h in CONTROL_HINTS if len(h) >= 4):
            return g
    return groups[0]


def has_hint(name, hints) -> bool:
    low = str(name).lower().replace("_", "-").replace(" ", "")
    return any(h in low for h in hints)


def fmt(df: pd.DataFrame, digits: int = 3) -> pd.DataFrame:
    out = df.copy()
    for c in out.select_dtypes(include="number").columns:
        out[c] = out[c].round(digits)
    return out


def build_excel(sheets: dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, df in sheets.items():
            if df is None or df.empty:
                continue
            df.to_excel(writer, sheet_name=name[:31], index=False)
            ws = writer.sheets[name[:31]]
            for col_cells in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col_cells)
                ws.column_dimensions[col_cells[0].column_letter].width = min(max(10, width + 2), 80)
    return buf.getvalue()


def stop_with(msg: str):
    st.error(msg)
    st.stop()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"### 🧬 {APP_NAME}")
    st.markdown(
        "**Steps**\n"
        "1. Load Ct values or an instrument export (QuantStudio, CFX)\n"
        "2. Choose normalisation and control group\n"
        "3. Review replicate and sample quality\n"
        "4. Download results, figure, methods and results text"
    )
    st.markdown(
        "**Built for miRNA research**\n"
        "- Hemolysis check (miR-23a / miR-451a)\n"
        "- Spike-in check (e.g. cel-miR-39)\n"
        "- Global mean normalisation for panels\n"
        "- Sample-size planner for your next study"
    )
    st.divider()
    st.markdown(
        "**Need help with your data?**  \n"
        "Custom qPCR, miRNA or omics analysis, or a private app built for your lab."
    )
    st.link_button(CONTACT_LABEL, CONTACT_URL)
    st.divider()
    st.caption(
        "This app does not save your data; files are processed only for your session. "
        f"Built by {AUTHOR}." + (f" [Source code]({GITHUB_URL})" if GITHUB_URL else "")
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title(APP_NAME)
st.markdown(
    "Most labs still calculate ΔΔCt in Excel, where one wrong cell changes the result, and few check "
    "plasma samples for hemolysis before trusting a miRNA result. Upload your Ct values or the export "
    "file from your instrument and get **quality checks, fold changes, statistics, a publication-ready "
    "figure and draft methods and results text** in about a minute, with no coding."
)

# ---------------------------------------------------------------------------
# Step 1: Data
# ---------------------------------------------------------------------------
st.header("1. Load your data")
source = st.radio("Data source", list(EXAMPLES) + [UPLOAD], horizontal=True)
is_example = source in EXAMPLES
is_plasma_example = source == "Example: plasma miRNA panel (QuantStudio export)"

with st.expander("Accepted files and template"):
    st.markdown(
        "- **Instrument exports:** QuantStudio / 7500 / StepOne results files (.txt, .csv or .xlsx) and "
        "Bio-Rad CFX *Quantification Cq Results* (.csv or .xlsx). Header lines are skipped and NTC, "
        "standard and control wells are removed automatically.\n"
        "- **Your own table:** one row per well with **Sample**, **Target** and **Ct/Cq**, plus an optional "
        "**Group** column. Without a Group column, groups are guessed from sample names (e.g. *HCC_03* → *HCC*) "
        "and you can edit them.\n"
        "- 'Undetermined', 'N/A' or empty Ct cells count as no amplification. Decimal commas are fine."
    )
    st.dataframe(pd.read_csv(TEMPLATE_FILE), hide_index=True)
    with open(TEMPLATE_FILE, "rb") as fh:
        st.download_button("Download template (CSV)", fh.read(), "qpcr_template.csv", "text/csv")

if is_example:
    path = EXAMPLES[source]
    with open(path, "rb") as fh:
        file_bytes, file_name = fh.read(), path
    if is_plasma_example:
        st.info(
            "**Example data (simulated):** a 16-miRNA plasma panel in Healthy and HCC samples (8 each, "
            "duplicate wells) exported from a QuantStudio-style instrument, with the cel-miR-39 spike-in. "
            "It contains one hemolysed sample, one failed extraction, NTC wells and a low-abundance miRNA, "
            "so you can see every quality check at work."
        )
    else:
        st.info(
            "**Example data (simulated):** a liver miRNA panel in Normal, Cirrhosis and HCC tissue, "
            "6 samples per group in triplicate, with RNU48, miR-16-5p and U6 as candidate references. "
            "It includes one outlier replicate and one undetermined well."
        )
else:
    upload = st.file_uploader("Upload CSV, TXT or Excel", type=["csv", "txt", "tsv", "xlsx", "xlsm"])
    if upload is None:
        st.stop()
    file_bytes, file_name = upload.getvalue(), upload.name

try:
    raw, import_report = load_table(file_bytes, file_name)
except Exception as exc:  # noqa: BLE001
    stop_with(f"Could not read the file: {exc}")

if import_report["format"] != "Generic table" or import_report["control_wells_removed"]:
    st.success(
        f"Detected **{import_report['format']}** format: {import_report['wells_in_file']} wells, "
        f"{import_report['control_wells_removed']} NTC/standard/control wells removed."
    )

with st.expander("Preview data", expanded=not is_example):
    st.dataframe(raw.head(50), hide_index=True)

guess = core.guess_columns(raw)
cols = list(raw.columns)
st.markdown("**Match your columns**")
c1, c2, c3, c4 = st.columns(4)
mapping = {}
for box, key in zip((c1, c3, c4), (core.SAMPLE, core.TARGET, core.CT)):
    default = cols.index(guess[key]) if guess.get(key) in cols else None
    mapping[key] = box.selectbox(key, cols, index=default, placeholder="Choose column")
group_options = [ASSIGN] + cols
group_default = group_options.index(guess[core.GROUP]) if guess.get(core.GROUP) in cols else 0
mapping[core.GROUP] = c2.selectbox(core.GROUP, group_options, index=group_default)

if mapping[core.GROUP] == ASSIGN:
    if not mapping[core.SAMPLE]:
        stop_with("Choose the Sample column first.")
    samples = list(dict.fromkeys(raw[mapping[core.SAMPLE]].astype(str).str.strip()))
    st.markdown("**Assign groups**: guessed from sample names; edit any cell if needed.")
    group_table = st.data_editor(
        pd.DataFrame({"Sample": samples, "Group": [core.guess_group(s) for s in samples]}),
        hide_index=True, disabled=["Sample"], key=f"groups_{file_name}_{len(samples)}",
        height=min(400, 38 + 35 * len(samples)),
    )
    lookup = dict(zip(group_table["Sample"], group_table["Group"].astype(str).str.strip()))
    raw = raw.copy()
    raw["__Group__"] = raw[mapping[core.SAMPLE]].astype(str).str.strip().map(lookup)
    mapping[core.GROUP] = "__Group__"

try:
    data = core.standardize(raw, mapping)
except ValueError as exc:
    stop_with(str(exc))

targets_all = list(dict.fromkeys(data[core.TARGET]))
groups_all = list(dict.fromkeys(data[core.GROUP]))
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Samples", data[core.SAMPLE].nunique())
m2.metric("Groups", len(groups_all))
m3.metric("Targets", len(targets_all))
m4.metric("Wells", len(data))
m5.metric("Undetermined wells", int(data["Undetermined"].sum()))

if len(groups_all) < 2:
    stop_with("At least two groups are needed (e.g. Control and Treated).")
if len(targets_all) < 2:
    stop_with("At least two targets are needed.")

spike_candidates = [t for t in targets_all if has_hint(t, SPIKE_HINTS)]

# ---------------------------------------------------------------------------
# Step 2: Settings
# ---------------------------------------------------------------------------
st.header("2. Analysis settings")
s1, s2 = st.columns([3, 2])
with s2:
    control = st.selectbox("Control / calibrator group", groups_all, index=groups_all.index(guess_control(groups_all)))
    group_order = st.multiselect("Groups to include (in plot order)", groups_all,
                                 default=[control] + [g for g in groups_all if g != control])
with s1:
    methods_list = list(core.NORMALISATION)
    default_method = "global_mean" if is_plasma_example else "reference"
    normalisation = st.radio(
        "Normalisation", methods_list, index=methods_list.index(default_method),
        format_func=lambda k: core.NORMALISATION[k], horizontal=True,
        help="Reference genes: endogenous controls such as RNU48, miR-16 or GAPDH. "
             "Spike-in: an exogenous control added before extraction (e.g. cel-miR-39), common for serum/plasma. "
             "Global mean: the mean of all miRNAs detected in every sample; recommended for panels of ~20+ miRNAs.",
    )
    candidates, global_max_ct, global_exclude = [], 35.0, []
    if normalisation == "reference":
        suggested = core.suggest_references(targets_all)
        candidates = st.multiselect(
            "Candidate reference genes (for the stability check)", targets_all,
            default=suggested if suggested else targets_all[-1:],
        )
        default_refs = ["RNU48", "miR-16-5p"] if source == "Example: tissue miRNA, 3 groups" else candidates[:2]
        references = st.multiselect(
            "Reference genes used for normalisation", candidates or targets_all,
            default=[r for r in default_refs if r in (candidates or targets_all)],
            help="Use the most stable genes from the Reference genes tab. Two or more is best practice (MIQE).",
        )
    elif normalisation == "spike_in":
        spike_choice = st.selectbox("Spike-in assay", targets_all,
                                    index=targets_all.index(spike_candidates[0]) if spike_candidates else 0)
        references = [spike_choice]
        st.caption("Spike-in normalisation corrects extraction and RT efficiency, but not differences in "
                   "the amount of starting material. Keep sample volumes identical.")
    else:
        g1, g2 = st.columns([1, 2])
        global_max_ct = g1.number_input("Include assays with Ct ≤", 25.0, 40.0, 35.0, 0.5,
                                        help="Assays must be detected at or below this Ct in every sample.")
        global_exclude = g2.multiselect(
            "Exclude from the global mean (spike-ins, small-RNA controls)", targets_all,
            default=[t for t in targets_all if has_hint(t, SPIKE_HINTS + SMALL_RNA_CONTROL_HINTS)],
        )
        references = []  # filled after QC

if normalisation != "global_mean" and not references:
    stop_with("Choose the normaliser (reference gene or spike-in).")
if control not in group_order:
    stop_with("The control group must be included.")

with st.expander("Advanced settings: QC thresholds, efficiencies, statistics"):
    a1, a2, a3 = st.columns(3)
    sd_threshold = a1.number_input("Max SD between technical replicates (cycles)", 0.1, 2.0, 0.5, 0.05)
    late_ct = a2.number_input("Flag Ct values above", 25.0, 45.0, 35.0, 0.5)
    remove_outlier = a3.checkbox(
        "Remove the outlying replicate when SD is too high (triplicates or more)", value=True)

    b1, b2 = st.columns(2)
    test = b1.selectbox("Statistical test (each group vs control)", list(core.TESTS),
                        format_func=lambda k: core.TESTS[k],
                        index=list(core.TESTS).index("dunnett") if len(group_order) >= 3 else 0)
    correction = b2.selectbox("Multiple-testing correction across targets", list(core.CORRECTIONS),
                              index=list(core.CORRECTIONS).index("bh") if len(targets_all) > 8 else 0,
                              format_func=lambda k: core.CORRECTIONS[k])

    h1, h2, h3 = st.columns(3)
    hemo_high = h1.number_input("Hemolysis ΔCq: high risk at ≥", 4.0, 12.0, 7.0, 0.5)
    hemo_border = h2.number_input("Hemolysis ΔCq: borderline at ≥", 2.0, 10.0, 5.0, 0.5)
    spike_tol = h3.number_input("Spike-in: flag samples differing from the median by more than (cycles)",
                                0.25, 5.0, 1.0, 0.25)

# ---------------------------------------------------------------------------
# Replicate QC and sample QC (all samples), then sample exclusion
# ---------------------------------------------------------------------------
data_used = data[data[core.GROUP].isin(group_order)]
qc_all = core.replicate_qc(data_used, sd_threshold=sd_threshold, late_ct=late_ct, remove_outlier=remove_outlier)

mir23a = core.find_assay(targets_all, ["mir-23a", "23a-3p"])
mir451a = core.find_assay(targets_all, ["mir-451", "451a"])
hemo = None
if mir23a and mir451a:
    hemo = core.hemolysis_check(qc_all, mir23a, mir451a, high=hemo_high, borderline=hemo_border)
spike_for_qc = (references[0] if normalisation == "spike_in" else (spike_candidates[0] if spike_candidates else None))
spike_tbl = core.spike_in_qc(qc_all, spike_for_qc, spike_tol) if spike_for_qc else None

flagged_samples = []
if hemo is not None:
    flagged_samples += hemo.loc[hemo["Hemolysis risk"] == "High", core.SAMPLE].tolist()
if spike_tbl is not None:
    flagged_samples += spike_tbl.loc[spike_tbl["Status"] != "OK", core.SAMPLE].tolist()
flagged_samples = list(dict.fromkeys(flagged_samples))

all_samples = list(dict.fromkeys(data_used[core.SAMPLE]))
exclude_samples = st.multiselect(
    "Exclude samples from the analysis",
    all_samples,
    default=flagged_samples if is_plasma_example else [],
    help="For example samples flagged in the Sample quality tab (hemolysis or failed extraction).",
)
if flagged_samples:
    st.caption("⚠️ Flagged in **Sample quality**: " + ", ".join(flagged_samples)
               + ". Review them and decide whether to exclude.")

qc = qc_all[~qc_all[core.SAMPLE].isin(exclude_samples)]
if qc[core.GROUP].nunique() < 2:
    stop_with("Too many samples excluded: at least two groups must remain.")

if normalisation == "global_mean":
    references = core.global_mean_genes(qc, global_max_ct, exclude=global_exclude)
    if len(references) < 2:
        stop_with("Fewer than two assays are detected in every sample. Raise the Ct cut-off or use another method.")

if normalisation == "global_mean":
    target_choices = [t for t in targets_all if t not in global_exclude]
    default_targets = target_choices
elif normalisation == "spike_in":
    target_choices = [t for t in targets_all if t not in references]
    default_targets = [t for t in target_choices if not has_hint(t, SPIKE_HINTS + SMALL_RNA_CONTROL_HINTS)]
else:
    target_choices = [t for t in targets_all if t not in references]
    default_targets = [t for t in target_choices if t not in candidates]
targets = st.multiselect("Targets to report", target_choices, default=default_targets or target_choices)
if not targets:
    stop_with("Choose at least one target to report.")

eff_genes = list(dict.fromkeys((references if normalisation != "global_mean" else []) + targets))
with st.expander("Amplification efficiency per assay (default 100 %)"):
    st.caption("Use the Efficiency calculator tab if you ran a standard curve. "
               "With global mean normalisation every assay in the mean uses its listed efficiency.")
    eff_genes_all = list(dict.fromkeys(eff_genes + references))
    eff_table = st.data_editor(
        pd.DataFrame({"Gene": eff_genes_all, "Efficiency %": [100.0] * len(eff_genes_all)}),
        hide_index=True, disabled=["Gene"],
        column_config={"Efficiency %": st.column_config.NumberColumn(min_value=50.0, max_value=150.0, step=0.1)},
        key="eff_" + str(hash(tuple(eff_genes_all))),
    )
    efficiencies = dict(zip(eff_table["Gene"], eff_table["Efficiency %"].fillna(100.0)))

# ---------------------------------------------------------------------------
# Calculations
# ---------------------------------------------------------------------------
group_order_used = [g for g in group_order if g in set(qc[core.GROUP])]
try:
    results = core.relative_quantities(qc, references, control, efficiencies, report_targets=targets)
except ValueError as exc:
    stop_with(str(exc))
summary = core.group_summary(results, group_order_used)
comparisons = core.compare_groups(results, control, test=test, correction=correction)
nan_samples = results[results["Fold change"].isna()][[core.SAMPLE, core.TARGET]]
hemolysis_assessed = hemo is not None

# ---------------------------------------------------------------------------
# Step 3: Results
# ---------------------------------------------------------------------------
st.header("3. Results")
tabs = st.tabs([
    "Replicate QC", "Sample quality", "Normalisation", "Fold changes", "Statistics", "Figure",
    "Methods, results & export", "Plan next experiment", "Efficiency calculator",
])
(tab_qc, tab_sample, tab_ref, tab_res, tab_stats, tab_fig, tab_export, tab_plan, tab_eff) = tabs

with tab_qc:
    flagged = qc_all[qc_all["Flags"] != ""]
    if flagged.empty:
        st.success("All technical replicate sets passed QC.")
    else:
        st.warning(f"{len(flagged)} of {len(qc_all)} replicate sets have flags. Review them before reporting.")
        st.dataframe(fmt(flagged), hide_index=True)
    if not nan_samples.empty:
        st.error(f"{len(nan_samples)} sample/target combinations could not be calculated "
                 "(missing target or normaliser Ct). They are excluded from statistics.")
        st.dataframe(nan_samples, hide_index=True)
    with st.expander("All replicate sets"):
        st.dataframe(fmt(qc_all), hide_index=True)

with tab_sample:
    st.subheader("Hemolysis (serum / plasma)")
    if hemo is None:
        st.info("Include **miR-23a-3p** and **miR-451a** in your panel to screen serum or plasma samples for "
                "hemolysis. Red blood cells release miR-451a, which can distort circulating miRNA results.")
    else:
        n_high = int((hemo["Hemolysis risk"] == "High").sum())
        n_border = int((hemo["Hemolysis risk"] == "Borderline").sum())
        msg = (f"ΔCq = Cq({mir23a}) − Cq({mir451a}). {n_high} sample(s) at high risk (ΔCq ≥ {hemo_high:g}), "
               f"{n_border} borderline (≥ {hemo_border:g}).")
        (st.error if n_high else st.warning if n_border else st.success)(msg)
        st.dataframe(fmt(hemo, 2), hide_index=True)
        st.pyplot(plots.sample_qc_plot(hemo, "ΔCq (miR-23a − miR-451a)", group_order,
                                       lines={f"High risk (≥ {hemo_high:g})": hemo_high,
                                              f"Borderline (≥ {hemo_border:g})": hemo_border},
                                       ylabel="ΔCq miR-23a − miR-451a",
                                       highlight=hemo.loc[hemo["Hemolysis risk"].isin(["High", "Borderline"]),
                                                          core.SAMPLE].tolist()))
        st.caption("Thresholds follow Blondal et al., Methods 2013, and can be changed under Advanced settings.")

    st.subheader("Spike-in recovery")
    if spike_tbl is None:
        st.info("No spike-in assay detected (e.g. cel-miR-39, cel-miR-54, UniSp). If you add a fixed amount of "
                "spike-in before RNA extraction, include it to check extraction and RT efficiency per sample.")
    else:
        bad = spike_tbl[spike_tbl["Status"] != "OK"]
        if bad.empty:
            st.success(f"{spike_for_qc}: all samples within ±{spike_tol:g} cycles of the median.")
        else:
            st.error(f"{spike_for_qc}: {len(bad)} sample(s) differ from the median by more than {spike_tol:g} cycles "
                     "(possible extraction failure or RT inhibition): " + ", ".join(bad[core.SAMPLE]))
        st.dataframe(fmt(spike_tbl, 2), hide_index=True)
        med = float(spike_tbl[f"Cq {spike_for_qc}"].median())
        st.pyplot(plots.sample_qc_plot(spike_tbl, f"Cq {spike_for_qc}", group_order,
                                       lines={"Median": med, f"Median +{spike_tol:g}": med + spike_tol,
                                              f"Median −{spike_tol:g}": med - spike_tol},
                                       ylabel=f"Cq {spike_for_qc}",
                                       highlight=bad[core.SAMPLE].tolist()))
    if exclude_samples:
        st.info("Excluded from the analysis: " + ", ".join(exclude_samples))

genorm_final = genorm_steps = None
with tab_ref:
    if normalisation == "global_mean":
        st.markdown(f"**Global mean normalisation** using **{len(references)} assays** detected at Ct ≤ "
                    f"{global_max_ct:g} in every sample (Mestdagh et al., 2009).")
        st.write(", ".join(references))
        if len(references) < 10:
            st.warning("Global mean normalisation is most reliable with larger panels (roughly 20 or more "
                       "miRNAs). With fewer assays, consider reference genes instead.")
        left_out = [t for t in targets_all if t not in references and t not in global_exclude]
        if left_out:
            st.caption("Not detected in every sample, so not in the mean: " + ", ".join(left_out))
    elif normalisation == "spike_in":
        st.markdown(f"**Spike-in normalisation** to **{references[0]}**. See the Sample quality tab for its "
                    "recovery in each sample.")
    elif len(candidates) < 2:
        st.info("Select at least two candidate reference genes to compare their stability.")
    else:
        try:
            genorm_final, genorm_steps = core.genorm_ranking(qc, candidates, efficiencies)
        except ValueError as exc:
            st.warning(str(exc))
        if genorm_final is not None:
            best = genorm_final["Gene"].iloc[:2].tolist()
            st.markdown(
                f"**Most stable pair: {best[0]} and {best[1]}.** geNorm M measures how consistently a gene "
                "tracks the other candidates across samples (lower is better; ≤ 0.5 is typical for "
                "homogeneous samples, ≤ 1.0 for heterogeneous tissue)."
            )
            st.dataframe(fmt(genorm_final), hide_index=True)
            if sorted(best) != sorted(references):
                st.info(f"You are normalising to {', '.join(references)}. "
                        f"The geNorm analysis suggests {best[0]} + {best[1]}.")
            with st.expander("geNorm elimination steps"):
                st.dataframe(fmt(genorm_steps), hide_index=True)
        st.markdown("**Raw Ct by group.** A good reference gene shows no systematic difference between groups.")
        st.dataframe(fmt(core.reference_ct_spread(qc, candidates)), hide_index=True)
        st.pyplot(plots.reference_ct_plot(qc, candidates, group_order_used))

with tab_res:
    st.markdown(
        f"Fold change relative to **{control}** (= 1). Geometric means are the recommended summary for qPCR "
        "because fold changes are ratios."
    )
    show_cols = [core.TARGET, core.GROUP, "n", "Fold change (geometric mean)", "95% CI low", "95% CI high",
                 "Mean log2 FC", "SEM log2 FC", "Fold change (arithmetic mean)", "SEM (arithmetic)"]
    st.dataframe(fmt(summary[show_cols]), hide_index=True)
    with st.expander("Per-sample values (ΔCt, ΔΔCt, fold change)"):
        st.dataframe(fmt(results), hide_index=True)

with tab_stats:
    st.markdown(
        f"Each group is compared with **{control}** on the log2 scale (−ΔΔCt) using "
        f"**{core.TESTS[test]}**, with **{core.CORRECTIONS[correction]}** correction across targets."
    )
    if comparisons.empty:
        st.info("Not enough samples for statistics (need ≥ 2 per group).")
    else:
        stat_cols = [core.TARGET, "Comparison", "n (group)", "n (control)", "Fold change", "p-value",
                     "Adjusted p", "Significance"]
        if len(group_order_used) >= 3:
            stat_cols.append("Omnibus p (all groups)")
        st.dataframe(fmt(comparisons[stat_cols], 4), hide_index=True)
        small = comparisons[(comparisons["n (group)"] < 3) | (comparisons["n (control)"] < 3)]
        if not small.empty:
            st.warning("Some groups have fewer than 3 biological replicates; p-values will be unreliable.")

with tab_fig:
    f1, f2, f3, f4 = st.columns(4)
    scale = f1.radio("Y axis", ["Fold change", "log2 fold change"])
    err = f2.radio("Error bars", ["95% CI", "SEM"])
    show_pts = f3.checkbox("Show individual samples", value=True)
    show_stars = f3.checkbox("Show significance", value=True)
    fig_title = f4.text_input("Title", "")
    significant = [] if comparisons.empty else list(dict.fromkeys(
        comparisons.loc[comparisons["Adjusted p"] < 0.05, core.TARGET]))
    default_plot = [t for t in targets if t in significant][:8] if len(targets) > 8 and significant else targets[:12]
    plot_targets = st.multiselect("Targets in figure (in order)", targets, default=default_plot,
                                  help="With many targets, the significant ones are shown by default.")
    if plot_targets:
        fig = plots.fold_change_plot(
            results, summary, comparisons if show_stars else None, group_order_used, plot_targets,
            scale="log2" if scale.startswith("log2") else "linear",
            error="ci" if err == "95% CI" else "sem",
            show_points=show_pts, title=fig_title,
        )
        st.pyplot(fig)
        d1, d2, d3 = st.columns(3)
        d1.download_button("Download PNG (300 dpi)", plots.to_bytes(fig, "png"), "qpcr_figure.png", "image/png")
        d2.download_button("Download SVG (editable)", plots.to_bytes(fig, "svg"), "qpcr_figure.svg", "image/svg+xml")
        d3.download_button("Download PDF", plots.to_bytes(fig, "pdf"), "qpcr_figure.pdf", "application/pdf")

methods = core.methods_text(references, control, efficiencies, test, correction, sd_threshold, remove_outlier,
                            len(group_order_used), normalisation=normalisation, global_max_ct=global_max_ct,
                            hemolysis_assessed=hemolysis_assessed)
results_par = core.results_text(summary, comparisons, control, adjusted=correction != "none")
reference_list = core.references_for(normalisation, test, hemolysis_assessed)

with tab_export:
    st.markdown("**Methods paragraph**: check it matches your experiment, then paste it into your manuscript.")
    st.text_area("Methods", methods, height=170, label_visibility="collapsed")
    st.markdown("**Results paragraph (draft)**: generated from the statistics table. Edit and add biological context.")
    st.text_area("Results", results_par, height=170, label_visibility="collapsed")
    if exclude_samples:
        st.caption("Remember to report the excluded samples and why: " + ", ".join(exclude_samples))
    st.markdown("**References**")
    for ref in reference_list:
        st.markdown(f"- {ref}")

    settings = pd.DataFrame({
        "Setting": ["Date", "Normalisation", "Normaliser assays", "Control group", "Groups", "Targets",
                    "Excluded samples", "Replicate SD threshold", "Outlier removal", "Late Ct flag", "Test",
                    "Correction", "Efficiencies (%)"],
        "Value": [str(date.today()), core.NORMALISATION[normalisation], ", ".join(references), control,
                  ", ".join(group_order_used), ", ".join(targets), ", ".join(exclude_samples) or "None",
                  sd_threshold, "Yes" if remove_outlier else "No", late_ct, core.TESTS[test],
                  core.CORRECTIONS[correction], "; ".join(f"{k}: {v:g}" for k, v in efficiencies.items())],
    })
    sheets = {
        "Settings": settings,
        "Group summary": fmt(summary, 5),
        "Statistics": fmt(comparisons, 6) if not comparisons.empty else None,
        "Per-sample results": fmt(results, 5),
        "Replicate QC": fmt(qc_all, 4),
        "Hemolysis": fmt(hemo, 3) if hemo is not None else None,
        "Spike-in": fmt(spike_tbl, 3) if spike_tbl is not None else None,
        "Reference stability": fmt(genorm_final, 4) if genorm_final is not None else None,
        "Methods and results": pd.DataFrame({"Text": [methods, results_par] + reference_list}),
        "Input data": data,
    }
    st.download_button(
        "Download full report (Excel)", build_excel(sheets), "qpcr_results.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary",
    )

with tab_plan:
    st.markdown(
        "How many biological replicates will your next experiment need? This uses the variability "
        "measured in **your own data** (pooled SD of log2 fold change within groups) and a two-sided t-test."
    )
    p1, p2, p3 = st.columns(3)
    alpha = p1.selectbox("Significance level (α)", [0.05, 0.01], index=0)
    power = p2.selectbox("Power", [0.8, 0.9], index=0, format_func=lambda x: f"{int(x * 100)} %")
    fcs = p3.multiselect("Fold changes to detect", [1.25, 1.5, 2.0, 3.0, 4.0], default=[1.5, 2.0])
    if fcs:
        plan = core.sample_size_table(results, fold_changes=tuple(sorted(fcs)), alpha=alpha, power=power)
        st.dataframe(fmt(plan, 3), hide_index=True)
        st.caption("These are estimates for planning; add a few extra samples to allow for failed extractions "
                   "or outliers. Use them in thesis proposals and grant applications.")

with tab_eff:
    st.markdown(
        "Paste a dilution series for one assay to calculate its amplification efficiency. "
        "Enter the relative quantity (e.g. 1, 0.1, 0.01) or tick the box if you already have log10 values."
    )
    default_curve = pd.DataFrame({"Quantity": [1, 0.1, 0.01, 0.001, 0.0001],
                                  "Ct": [18.1, 21.45, 24.8, 28.2, 31.5]})
    curve_df = st.data_editor(default_curve, num_rows="dynamic", hide_index=True, key="curve")
    is_log = st.checkbox("Quantities are already log10")
    try:
        sc = core.standard_curve(curve_df["Quantity"], curve_df["Ct"], log_input=is_log)
        e1, e2, e3 = st.columns(3)
        e1.metric("Efficiency", f"{sc.efficiency_percent:.1f} %")
        e2.metric("Slope", f"{sc.slope:.3f}")
        e3.metric("R²", f"{sc.r_squared:.4f}")
        for note in sc.notes:
            st.warning(note)
        xq = np.asarray(curve_df["Quantity"], dtype=float)
        st.pyplot(plots.standard_curve_plot(xq if is_log else np.log10(xq), curve_df["Ct"].to_numpy(dtype=float), sc))
        st.caption("Enter this efficiency for the assay in the efficiency table above.")
    except (ValueError, TypeError) as exc:
        st.info(str(exc))

st.divider()
st.caption(
    "Methods: Livak & Schmittgen 2001; Hellemans et al. 2007; Vandesompele et al. 2002; Mestdagh et al. 2009; "
    "Blondal et al. 2013. Calculations are checked by automated tests against independent implementations. "
    f"Always review quality flags before publishing. © {date.today().year} {AUTHOR}"
)
