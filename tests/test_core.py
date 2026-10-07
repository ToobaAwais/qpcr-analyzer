"""
Tests for core.py. Run with:  pytest -q

Every calculation is checked against an independent re-implementation,
a hand-worked example, or published reference values (R's p.adjust).
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import core  # noqa: E402

MAP = {core.SAMPLE: "Sample", core.GROUP: "Group", core.TARGET: "Target", core.CT: "Ct"}


def make(rows):
    """rows: (sample, group, target, ct)"""
    return core.standardize(pd.DataFrame(rows, columns=["Sample", "Group", "Target", "Ct"]), MAP)


@pytest.fixture(scope="module")
def example():
    raw = pd.read_csv(ROOT / "data" / "example_mirna_qpcr.csv")
    data = core.standardize(raw, core.guess_columns(raw))
    qc = core.replicate_qc(data, remove_outlier=True)
    return data, qc


# ---------------------------------------------------------------- ΔΔCt -----

def test_livak_hand_worked_example():
    # Control: target 25, ref 20 -> ΔCt 5.  Treated: target 23, ref 20 -> ΔCt 3.
    # ΔΔCt = 3 - 5 = -2  ->  fold change 2^2 = 4
    d = make([("C1", "Ctrl", "miR-X", 25.0), ("C1", "Ctrl", "U6", 20.0),
              ("T1", "Trt", "miR-X", 23.0), ("T1", "Trt", "U6", 20.0)])
    res = core.relative_quantities(core.replicate_qc(d), ["U6"], "Ctrl")
    t = res.set_index("Sample").loc["T1"]
    assert t["ΔCt"] == pytest.approx(3.0)
    assert t["ΔΔCt"] == pytest.approx(-2.0)
    assert t["Fold change"] == pytest.approx(4.0)
    assert res.set_index("Sample").loc["C1", "Fold change"] == pytest.approx(1.0)


def test_matches_independent_livak_on_example(example):
    _, qc = example
    refs = ["RNU48", "miR-16-5p"]
    res = core.relative_quantities(qc, refs, "Normal")

    # Independent implementation of 2^-ΔΔCt
    wide = qc.pivot_table(index=["Sample", "Group"], columns="Target", values="Mean Ct")
    ref_mean = wide[refs].mean(axis=1)
    for target in ["miR-21-5p", "miR-122-5p", "miR-221-3p", "miR-199a-3p", "U6"]:
        dct = wide[target] - ref_mean
        ctrl_mean = dct[dct.index.get_level_values("Group") == "Normal"].mean()
        ddct = dct - ctrl_mean
        got = res[res.Target == target].set_index(["Sample", "Group"])["ΔΔCt"]
        pd.testing.assert_series_equal(got.sort_index(), ddct.sort_index(), check_names=False, atol=1e-10)


def test_recovers_exact_fold_change_despite_rna_loading_differences():
    # Each sample has a different RNA input (shifts every gene equally);
    # normalisation must remove it and recover the true 8-fold change.
    rows, rng = [], np.random.default_rng(1)
    for i in range(4):
        for grp, shift in (("Ctrl", 0.0), ("Trt", -3.0)):  # -3 cycles = 8x more target
            load = rng.normal(0, 1.5)
            s = f"{grp}{i}"
            rows += [(s, grp, "T", 26 + shift + load), (s, grp, "R1", 22 + load), (s, grp, "R2", 24 + load)]
    res = core.relative_quantities(core.replicate_qc(make(rows)), ["R1", "R2"], "Ctrl")
    trt = res[res.Group == "Trt"]["Fold change"]
    assert np.allclose(trt, 8.0)


def test_efficiency_correction_matches_pfaffl():
    # Pfaffl (2001): ratio = E_target^ΔCt_target(control - sample) / E_ref^ΔCt_ref(control - sample)
    e_t, e_r = 1.92, 2.05  # 92 % and 105 %
    d = make([("C", "Ctrl", "T", 24.0), ("C", "Ctrl", "R", 18.0),
              ("S", "Trt", "T", 21.5), ("S", "Trt", "R", 18.4)])
    res = core.relative_quantities(core.replicate_qc(d), ["R"], "Ctrl",
                                   efficiencies={"T": 92.0, "R": 105.0})
    expected = e_t ** (24.0 - 21.5) / e_r ** (18.0 - 18.4)
    assert res.set_index("Sample").loc["S", "Fold change"] == pytest.approx(expected)


def test_100_percent_efficiency_equals_livak():
    d = make([("C", "Ctrl", "T", 24.0), ("C", "Ctrl", "R", 18.0),
              ("S", "Trt", "T", 21.5), ("S", "Trt", "R", 18.4)])
    qc = core.replicate_qc(d)
    a = core.relative_quantities(qc, ["R"], "Ctrl")
    b = core.relative_quantities(qc, ["R"], "Ctrl", efficiencies={"T": 100, "R": 100})
    assert a["Fold change"].tolist() == pytest.approx(b["Fold change"].tolist())
    assert a.set_index("Sample").loc["S", "Fold change"] == pytest.approx(2 ** -((21.5 - 18.4) - (24 - 18)))


def test_missing_reference_gives_nan_not_a_wrong_answer():
    d = make([("C", "Ctrl", "T", 24.0), ("C", "Ctrl", "R1", 18.0), ("C", "Ctrl", "R2", 20.0),
              ("S", "Trt", "T", 22.0), ("S", "Trt", "R1", 18.0), ("S", "Trt", "R2", "Undetermined")])
    res = core.relative_quantities(core.replicate_qc(d), ["R1", "R2"], "Ctrl")
    assert np.isnan(res.set_index("Sample").loc["S", "Fold change"])


def test_example_recovers_simulated_truth(example):
    _, qc = example
    res = core.relative_quantities(qc, ["RNU48", "miR-16-5p"], "Normal")
    summ = core.group_summary(res)
    truth = pd.read_csv(ROOT / "data" / "example_truth.csv")
    merged = summ.merge(truth, on=["Target", "Group"])
    assert len(merged) == 12
    # estimates fall inside their own 95 % CI band around truth +/- noise
    assert (merged["Mean log2 FC"] - merged["True log2 FC"]).abs().max() < 0.7
    hcc = merged[merged.Group == "HCC"].set_index("Target")["Mean log2 FC"]
    assert hcc["miR-21-5p"] > 1 and hcc["miR-221-3p"] > 1
    assert hcc["miR-122-5p"] < -1 and hcc["miR-199a-3p"] < -1


# ------------------------------------------------------- replicate QC -----

def test_outlier_removal_drops_replicate_furthest_from_median():
    d = make([("S", "G", "T", 25.0), ("S", "G", "T", 25.1), ("S", "G", "T", 27.0)])
    kept = core.replicate_qc(d, remove_outlier=True).iloc[0]
    assert kept["Mean Ct"] == pytest.approx(25.05)
    assert kept["Used"] == 2 and "Outlier removed" in kept["Flags"]
    flagged = core.replicate_qc(d, remove_outlier=False).iloc[0]
    assert flagged["Mean Ct"] == pytest.approx((25 + 25.1 + 27) / 3)
    assert "High SD" in flagged["Flags"]


def test_undetermined_and_decimal_comma_parsing():
    raw = pd.DataFrame({"Sample": ["A"] * 4, "Group": ["G"] * 4, "Target": ["T"] * 4,
                        "Ct": ["Undetermined", "", "24,5", "24.7"]})
    d = core.standardize(raw, MAP)
    assert d["Undetermined"].tolist() == [True, True, False, False]
    q = core.replicate_qc(d).iloc[0]
    assert q["Mean Ct"] == pytest.approx(24.6)
    assert "2 undetermined" in q["Flags"]


def test_late_ct_and_no_amplification_flags():
    d = make([("A", "G", "T", 36.0), ("A", "G", "T", 36.2), ("B", "G", "T", "Undetermined")])
    qc = core.replicate_qc(d).set_index("Sample")
    assert "Late Ct" in qc.loc["A", "Flags"]
    assert "No amplification" in qc.loc["B", "Flags"]


def test_sample_in_two_groups_is_rejected():
    with pytest.raises(ValueError, match="more than one group"):
        make([("A", "G1", "T", 20.0), ("A", "G2", "R", 20.0)])


def test_column_guessing_handles_instrument_style_names():
    df = pd.DataFrame(columns=["Sample Name", "Condition", "Target Name", "Cq"])
    g = core.guess_columns(df)
    assert g == {"Sample": "Sample Name", "Group": "Condition", "Target": "Target Name", "Ct": "Cq"}


# ------------------------------------------------------------ stats -----

def test_p_adjust_matches_r_reference_values():
    p = [0.01, 0.02, 0.03, 0.04, 0.05]
    # R: p.adjust(p, "holm") ; p.adjust(p, "BH")
    assert list(core.p_adjust(p, "holm")) == pytest.approx([0.05, 0.08, 0.09, 0.09, 0.09])
    assert list(core.p_adjust(p, "bh")) == pytest.approx([0.05] * 5)
    p2 = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205]
    assert list(core.p_adjust(p2, "bh")) == pytest.approx(
        [0.008, 0.032, 0.0672, 0.0672, 0.0672, 0.08, 0.0845714286, 0.205])
    shuffled = [0.05, 0.01, 0.04, 0.02, 0.03]
    assert list(core.p_adjust(shuffled, "holm")) == pytest.approx([0.09, 0.05, 0.09, 0.08, 0.09])
    assert np.isnan(core.p_adjust([0.01, np.nan], "holm")[1])


@pytest.mark.parametrize("test,fn", [
    ("welch", lambda a, b: stats.ttest_ind(a, b, equal_var=False).pvalue),
    ("student", lambda a, b: stats.ttest_ind(a, b, equal_var=True).pvalue),
    ("mannwhitney", lambda a, b: stats.mannwhitneyu(a, b, alternative="two-sided").pvalue),
])
def test_pvalues_match_scipy_on_minus_ddct(example, test, fn):
    _, qc = example
    res = core.relative_quantities(qc, ["RNU48", "miR-16-5p"], "Normal")
    out = core.compare_groups(res, "Normal", test=test, correction="none")
    for _, row in out.iterrows():
        sub = res[res.Target == row["Target"]]
        a = -sub.loc[sub.Group == row["Group"], "ΔΔCt"].to_numpy()
        b = -sub.loc[sub.Group == "Normal", "ΔΔCt"].to_numpy()
        assert row["p-value"] == pytest.approx(fn(a, b), rel=1e-12)


def test_omnibus_anova_matches_scipy(example):
    _, qc = example
    res = core.relative_quantities(qc, ["RNU48", "miR-16-5p"], "Normal")
    out = core.compare_groups(res, "Normal")
    sub = res[res.Target == "miR-21-5p"]
    expected = stats.f_oneway(*[sub.loc[sub.Group == g, "log2 FC"] for g in ["Normal", "Cirrhosis", "HCC"]]).pvalue
    got = out[out.Target == "miR-21-5p"]["Omnibus p (all groups)"].iloc[0]
    assert got == pytest.approx(expected)


def test_group_summary_ci_is_t_based():
    res = pd.DataFrame({"Sample": list("abcd"), "Group": ["G"] * 4, "Target": ["T"] * 4,
                        "log2 FC": [1.0, 2.0, 3.0, 2.0]})
    res["Fold change"] = 2 ** res["log2 FC"]
    s = core.group_summary(res).iloc[0]
    sem = np.std([1, 2, 3, 2], ddof=1) / 2
    assert s["Fold change (geometric mean)"] == pytest.approx(4.0)
    assert s["95% CI high"] == pytest.approx(2 ** (2 + stats.t.ppf(0.975, 3) * sem))


# ----------------------------------------------------------- geNorm -----

def naive_genorm(log2q: pd.DataFrame) -> dict:
    """Straight transcription of Vandesompele 2002 eq. for M_j."""
    genes = list(log2q.columns)
    out = {}
    for j in genes:
        v = []
        for k in genes:
            if k == j:
                continue
            a = [log2q[j].iloc[i] - log2q[k].iloc[i] for i in range(len(log2q))]
            mu = sum(a) / len(a)
            v.append(math.sqrt(sum((x - mu) ** 2 for x in a) / (len(a) - 1)))
        out[j] = sum(v) / len(v)
    return out


def test_genorm_matches_naive_implementation():
    rng = np.random.default_rng(7)
    q = pd.DataFrame(rng.normal(0, 1, size=(10, 4)), columns=list("ABCD"))
    got = core.genorm_m(q)
    for g, m in naive_genorm(q).items():
        assert got[g] == pytest.approx(m)


def test_genorm_coregulated_genes_have_zero_m():
    base = pd.Series([1.0, 3.0, 2.0, 5.0])
    q = pd.DataFrame({"A": base, "B": base + 2.0})  # constant ratio
    assert core.genorm_m(q).tolist() == pytest.approx([0.0, 0.0])


def test_genorm_ranks_unstable_u6_last(example):
    _, qc = example
    final, steps = core.genorm_ranking(qc, ["U6", "RNU48", "miR-16-5p"])
    assert final["Gene"].iloc[-1] == "U6"
    assert len(steps) == 2


# --------------------------------------------------- standard curve -----

def test_standard_curve_perfect_efficiency():
    q = np.array([1e5, 1e4, 1e3, 1e2, 1e1])
    slope = -1 / math.log10(2)  # -3.3219 = 100 % efficiency
    ct = 35 + slope * np.log10(q)
    sc = core.standard_curve(q, ct)
    assert sc.efficiency_percent == pytest.approx(100.0)
    assert sc.r_squared == pytest.approx(1.0)
    assert sc.notes == []


def test_standard_curve_known_slope():
    sc = core.standard_curve([0, 1, 2, 3], [30, 26.2, 22.4, 18.6], log_input=True)  # slope -3.8
    assert sc.efficiency_percent == pytest.approx((10 ** (1 / 3.8) - 1) * 100)  # 83.3 %
    assert "outside" in sc.notes[0]


def test_methods_text_mentions_settings():
    txt = core.methods_text(["RNU48", "miR-16-5p"], "Normal", {"x": 100}, "welch", "holm", 0.5, True, 3)
    for word in ("RNU48", "Normal", "Welch", "Holm", "geometric mean", "ANOVA", "2^-ΔΔCt"):
        assert word in txt


# ------------------------------------------------- normalisation modes -----

def test_global_mean_uses_assays_detected_in_every_sample():
    rows = []
    for s, g in (("A", "C"), ("B", "C"), ("X", "T"), ("Y", "T")):
        rows += [(s, g, "m1", 25.0), (s, g, "m2", 27.0), (s, g, "m3", 29.0), (s, g, "spike", 22.0)]
    rows += [("A", "C", "rare", 34.0), ("B", "C", "rare", 36.5), ("X", "T", "rare", 33.0)]  # not in Y, >35 in B
    qc = core.replicate_qc(make(rows))
    genes = core.global_mean_genes(qc, max_ct=35, exclude=["spike"])
    assert genes == ["m1", "m2", "m3"]


def test_global_mean_normalisation_matches_hand_calculation():
    # Treated samples: m1 3 cycles earlier, others unchanged, plus a loading shift of +1 everywhere
    rows = []
    for s in ("C1", "C2"):
        rows += [(s, "C", "m1", 25.0), (s, "C", "m2", 27.0), (s, "C", "m3", 29.0)]
    for s in ("T1", "T2"):
        rows += [(s, "T", "m1", 23.0), (s, "T", "m2", 28.0), (s, "T", "m3", 30.0)]
    qc = core.replicate_qc(make(rows))
    genes = core.global_mean_genes(qc)
    res = core.relative_quantities(qc, genes, "C", report_targets=genes).set_index(["Sample", "Target"])
    # global mean Ct: control 27.0, treated 27.0 -> ΔCt(m1) = -4 (C: -2) -> ΔΔCt = -2 -> FC 4
    assert res.loc[("T1", "m1"), "ΔΔCt"] == pytest.approx(-2.0)
    assert res.loc[("T1", "m1"), "Fold change"] == pytest.approx(4.0)
    assert res.loc[("T1", "m2"), "Fold change"] == pytest.approx(0.5)


def test_report_targets_can_include_normaliser_genes():
    d = make([("C", "Ctrl", "a", 20.0), ("C", "Ctrl", "b", 22.0),
              ("T", "Trt", "a", 19.0), ("T", "Trt", "b", 22.0)])
    res = core.relative_quantities(core.replicate_qc(d), ["a", "b"], "Ctrl", report_targets=["a", "b"])
    assert set(res["Target"]) == {"a", "b"}


# ------------------------------------------------ circulating miRNA QC -----

def test_hemolysis_ratio_and_categories():
    rows = []
    for s, d451 in (("ok", 22.0), ("border", 20.5), ("hemo", 18.5)):
        rows += [(s, "G", "miR-23a-3p", 26.0), (s, "G", "hsa-miR-451a", d451)]
    qc = core.replicate_qc(make(rows))
    targets = qc["Target"].unique()
    assert core.find_assay(targets, ["23a"]) == "miR-23a-3p"
    assert core.find_assay(targets, ["451"]) == "hsa-miR-451a"
    h = core.hemolysis_check(qc, "miR-23a-3p", "hsa-miR-451a").set_index("Sample")
    assert h.loc["ok", "ΔCq (miR-23a − miR-451a)"] == pytest.approx(4.0)
    assert h.loc["ok", "Hemolysis risk"] == "Low"
    assert h.loc["border", "Hemolysis risk"] == "Borderline"
    assert h.loc["hemo", "Hemolysis risk"] == "High"


def test_spike_in_flags_failed_extraction():
    rows = [(s, "G", "cel-miR-39", ct) for s, ct in
            (("a", 23.0), ("b", 23.2), ("c", 22.9), ("d", 25.6), ("e", 23.1))]
    out = core.spike_in_qc(core.replicate_qc(make(rows)), "cel-miR-39").set_index("Sample")
    assert out.loc["d", "Status"] == "Check sample"
    assert (out.drop("d")["Status"] == "OK").all()
    assert out.loc["d", "Difference from median"] == pytest.approx(25.6 - 23.1)


def test_plasma_example_detects_planted_problems():
    rows = core.rows_from_text((ROOT / "data" / "example_plasma_quantstudio.txt").read_text())
    table, report = core.clean_instrument_table(core.find_table(rows))
    assert report["control_wells_removed"] == 17
    table["Group"] = table["Sample Name"].map(core.guess_group)
    mapping = core.guess_columns(table)
    mapping["Group"] = "Group"
    qc = core.replicate_qc(core.standardize(table, mapping))
    hemo = core.hemolysis_check(qc, "miR-23a-3p", "miR-451a").set_index("Sample")
    assert hemo.loc["Healthy_03", "Hemolysis risk"] == "High"
    spike = core.spike_in_qc(qc, "cel-miR-39-3p").set_index("Sample")
    assert list(spike[spike.Status != "OK"].index) == ["HCC_05"]
    genes = core.global_mean_genes(qc, 35, exclude=["cel-miR-39-3p"])
    assert "miR-375-3p" not in genes and len(genes) == 15
    res = core.relative_quantities(qc, genes, "Healthy", report_targets=genes)
    up = core.group_summary(res).set_index(["Target", "Group"]).loc[("miR-122-5p", "HCC"), "Mean log2 FC"]
    assert 2.0 < up < 3.2  # simulated +2.5 log2


# ------------------------------------------------- instrument imports -----

def test_quantstudio_style_export_is_parsed():
    text = "\n".join([
        "* Experiment Name = test", "* Instrument Type = QuantStudio", "", "[Results]",
        "Well\tWell Position\tSample Name\tTarget Name\tTask\tCT",
        "1\tA1\tCtrl_1\tmiR-21\tUNKNOWN\t24.1",
        "2\tA2\tCtrl_1\tU6\tUNKNOWN\t20.0",
        "3\tA3\tNTC\tmiR-21\tNTC\tUndetermined",
        "4\tA4\tStd1\tmiR-21\tSTANDARD\t18.0",
    ])
    t, rep = core.clean_instrument_table(core.find_table(core.rows_from_text(text)))
    assert rep["format"].startswith("Applied Biosystems")
    assert len(t) == 2 and rep["control_wells_removed"] == 2
    assert core.guess_columns(t)["Ct"] == "CT"


def test_biorad_cfx_style_export_is_parsed():
    text = "\n".join([
        ",Well,Fluor,Target,Content,Sample,Biological Set Name,Cq,Cq Mean",
        "0,A01,FAM,IL6,Unkn-01,Veh 1,,27.1,27.2",
        "1,A02,FAM,GAPDH,Unkn-01,Veh 1,,18.2,18.2",
        "2,A03,FAM,IL6,NTC,,,,",
        "3,A04,FAM,IL6,Std-01,,,15.0,15.0",
    ])
    t, rep = core.clean_instrument_table(core.find_table(core.rows_from_text(text)))
    assert rep["format"] == "Bio-Rad CFX"
    assert len(t) == 2
    g = core.guess_columns(t)
    assert g["Ct"] == "Cq" and g["Sample"] == "Sample" and g["Target"] == "Target"


@pytest.mark.parametrize("name,group", [
    ("NOR-03", "NOR"), ("Control 2", "Control"), ("HCC_12", "HCC"), ("Healthy_01", "Healthy"),
    ("KO.3", "KO"), ("WT", "WT"), ("Tumor-A1", "Tumor"),
])
def test_group_guessed_from_sample_name(name, group):
    assert core.guess_group(name) == group


# ---------------------------------------------------------- Dunnett -----

def test_dunnett_matches_scipy(example):
    _, qc = example
    res = core.relative_quantities(qc, ["RNU48", "miR-16-5p"], "Normal")
    out = core.compare_groups(res, "Normal", test="dunnett", correction="none")
    sub = res[res.Target == "miR-21-5p"]
    groups = [sub.loc[sub.Group == g, "log2 FC"].to_numpy() for g in ("Cirrhosis", "HCC")]
    ctrl = sub.loc[sub.Group == "Normal", "log2 FC"].to_numpy()
    expected = stats.dunnett(*groups, control=ctrl).pvalue
    got = out[out.Target == "miR-21-5p"].set_index("Group")["p-value"]
    # SciPy's Dunnett p-values are computed by quasi-Monte Carlo integration
    assert got["Cirrhosis"] == pytest.approx(expected[0], abs=2e-3)
    assert got["HCC"] == pytest.approx(expected[1], abs=2e-3)


# ---------------------------------------------------- sample size -----

def test_sample_size_matches_textbook_values():
    # Cohen's d = 1 -> 17 per group; d = 0.5 -> 64 per group (alpha 0.05, power 0.80, two-sided)
    assert core.n_per_group(1.0, 1.0) == 17
    assert core.n_per_group(0.5, 1.0) == 64
    assert core.t_test_power(17, 1.0, 1.0) == pytest.approx(0.807, abs=0.005)


def test_sample_size_table_uses_pooled_sd():
    res = pd.DataFrame({"Sample": list("abcdef"), "Group": ["C"] * 3 + ["T"] * 3, "Target": ["x"] * 6,
                        "log2 FC": [0.0, 0.5, 1.0, 1.0, 1.5, 2.0]})
    t = core.sample_size_table(res, fold_changes=(2.0,)).iloc[0]
    assert t["Pooled SD (log2 FC)"] == pytest.approx(0.5)
    assert t["n per group for 2-fold"] == core.n_per_group(1.0, 0.5)


# ------------------------------------------------- written outputs -----

def test_results_text_reports_direction_and_p(example):
    _, qc = example
    res = core.relative_quantities(qc, ["RNU48", "miR-16-5p"], "Normal")
    res = res[res.Target != "U6"]
    summ = core.group_summary(res)
    comp = core.compare_groups(res, "Normal")
    txt = core.results_text(summ, comp, "Normal")
    assert "miR-21-5p was significantly up-regulated in HCC compared with Normal" in txt
    assert "miR-122-5p was significantly down-regulated in HCC" in txt
    assert "adjusted p < 0.001" in txt


def test_methods_text_for_global_mean_and_hemolysis():
    txt = core.methods_text(["a"] * 15, "Healthy", {}, "dunnett", "bh", 0.5, False, 2,
                            normalisation="global_mean", hemolysis_assessed=True)
    assert "global mean of all 15 assays" in txt and "Mestdagh" in txt and "miR-23a/miR-451a" in txt
    refs = core.references_for("global_mean", "dunnett", True)
    assert any("Mestdagh" in r for r in refs) and any("Blondal" in r for r in refs) and any("Dunnett" in r for r in refs)
