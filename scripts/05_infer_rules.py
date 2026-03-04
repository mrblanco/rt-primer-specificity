"""
Infer rules of non-specific RT priming from complementarity analysis results.

Performs:
1. Per-primer feature correlation with off-target rate
2. Distribution analysis of complementarity metrics at off-target sites
3. Random genomic site negative control comparison
4. Logistic regression and random forest models
5. Comprehensive visualizations

Usage:
    python 05_infer_rules.py \
        --complementarity complementarity_analysis.tsv.gz \
        --primer-summary primer_summary.tsv \
        --primers ../data/rt_primers.csv \
        --output-dir ../results
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import helpers

warnings.filterwarnings("ignore", category=FutureWarning)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Infer rules of non-specific RT priming.",
    )
    parser.add_argument(
        "--complementarity", required=True,
        help="Path to complementarity_analysis.tsv.gz from step 04.",
    )
    parser.add_argument(
        "--primer-summary", required=True,
        help="Path to primer_summary.tsv from step 02.",
    )
    parser.add_argument(
        "--primers", required=True,
        help="Path to RT primer CSV file.",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Path to output directory for figures and reports.",
    )
    parser.add_argument(
        "--n-random-sites", type=int, default=50000,
        help="Number of random genomic sites to generate as negative controls.",
    )
    parser.add_argument(
        "--random-controls",
        help="Path to pre-computed random control complementarity TSV (optional). "
             "If not provided, random sequences are simulated with matched GC content.",
    )
    return parser.parse_args()


# ============================================================
# Primer feature computation
# ============================================================

def compute_primer_features(primers_df):
    """Compute sequence features for each primer."""
    features = []
    for _, row in primers_df.iterrows():
        seq = row["Seq"]
        length = len(seq)
        gc = sum(1 for b in seq if b in "GCgc") / length

        # 3' end features
        last_6 = seq[-6:]
        last_8 = seq[-8:]
        gc_3prime_6 = sum(1 for b in last_6 if b in "GCgc") / len(last_6)
        gc_3prime_8 = sum(1 for b in last_8 if b in "GCgc") / len(last_8)

        # Max homopolymer run
        max_homo = 1
        current_run = 1
        for i in range(1, length):
            if seq[i] == seq[i - 1]:
                current_run += 1
                max_homo = max(max_homo, current_run)
            else:
                current_run = 1

        # Dinucleotide entropy (sequence complexity)
        dinucs = [seq[i:i+2] for i in range(length - 1)]
        dinuc_counts = {}
        for d in dinucs:
            dinuc_counts[d] = dinuc_counts.get(d, 0) + 1
        total_dinucs = len(dinucs)
        entropy = 0.0
        for count in dinuc_counts.values():
            p = count / total_dinucs
            if p > 0:
                entropy -= p * np.log2(p)

        # 3' terminal base identity
        terminal_base = seq[-1]

        features.append({
            "probe_name": row["ProbeName"],
            "primer_length": length,
            "primer_gc": gc,
            "gc_3prime_6": gc_3prime_6,
            "gc_3prime_8": gc_3prime_8,
            "max_homopolymer": max_homo,
            "dinuc_entropy": round(entropy, 3),
            "terminal_base": terminal_base,
            "last_6mer": last_6,
            "dG37": row["dG37"],
        })

    return pd.DataFrame(features)


# ============================================================
# Random control sequence generation
# ============================================================

def generate_random_sequences(n_seqs, seq_length, gc_content=0.42):
    """Generate random DNA sequences with specified GC content."""
    rng = np.random.default_rng(42)
    seqs = []
    n_gc = int(seq_length * gc_content)
    n_at = seq_length - n_gc

    for _ in range(n_seqs):
        bases = (
            list(rng.choice(["G", "C"], size=n_gc)) +
            list(rng.choice(["A", "T"], size=n_at))
        )
        rng.shuffle(bases)
        seqs.append("".join(bases))

    return seqs


def compute_random_control_metrics(primer_seqs, n_random, seq_length=40):
    """
    Compute complementarity metrics for random genomic sequences against each primer.
    This serves as a null distribution.
    """
    from scripts_04 import analyze_site
    # Import analyze_site from step 04 — but since we're in the same directory,
    # we handle it inline to avoid circular imports
    from importlib import import_module

    records = []
    rng = np.random.default_rng(42)
    primers_list = list(primer_seqs.items())

    random_seqs = generate_random_sequences(n_random, seq_length)

    for seq in random_seqs:
        # Pick a random primer
        primer_name, primer_seq = primers_list[rng.integers(len(primers_list))]
        metrics = _analyze_site_inline(primer_seq, seq)
        metrics["rt_primer"] = primer_name
        metrics["is_offtarget"] = 0  # label for classification
        records.append(metrics)

    return pd.DataFrame(records)


def _analyze_site_inline(primer_seq, site_seq):
    """Inline version of complementarity analysis (avoids import issues)."""
    def _count_3p(p, s):
        count = 0
        for a, b in zip(reversed(p), reversed(s)):
            if a.upper() == b.upper():
                count += 1
            else:
                break
        return count

    def _matches_window(p, s, start, size):
        p_end = len(p)
        p_start = max(0, p_end - start - size)
        p_stop = p_end - start
        s_end = len(s)
        s_start = max(0, s_end - start - size)
        s_stop = s_end - start
        if p_start >= p_stop or s_start >= s_stop:
            return 0, 0
        pw = p[p_start:p_stop].upper()
        sw = s[s_start:s_stop].upper()
        ml = min(len(pw), len(sw))
        return sum(1 for a, b in zip(pw[-ml:], sw[-ml:]) if a == b), ml

    m6, w6 = _matches_window(primer_seq, site_seq, 0, 6)
    m8, w8 = _matches_window(primer_seq, site_seq, 0, 8)
    m10, w10 = _matches_window(primer_seq, site_seq, 0, 10)
    m12, w12 = _matches_window(primer_seq, site_seq, 0, 12)

    return {
        "match_3prime_consecutive": _count_3p(primer_seq, site_seq),
        "matches_last_6": m6,
        "matches_last_8": m8,
        "matches_last_10": m10,
        "matches_last_12": m12,
        "best_contiguous_match": m8,  # simplified for random controls
    }


# ============================================================
# Visualization functions
# ============================================================

def plot_ontarget_rates(summary, output_dir):
    """Bar chart of on-target rate per primer, colored by gene."""
    fig, ax = plt.subplots(figsize=(14, 6))

    summary_sorted = summary.sort_values("on_target_rate")
    genes = summary_sorted["gene_name"].unique()
    palette = dict(zip(genes, sns.color_palette("husl", len(genes))))
    colors = [palette[g] for g in summary_sorted["gene_name"]]

    bars = ax.bar(range(len(summary_sorted)), summary_sorted["on_target_rate"],
                  color=colors, edgecolor="black", linewidth=0.3)
    ax.set_xticks(range(len(summary_sorted)))
    ax.set_xticklabels(summary_sorted["rt_primer"], rotation=90, fontsize=6)
    ax.set_ylabel("On-target rate")
    ax.set_title("On-target rate per RT primer")
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)

    # Legend for genes
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=palette[g], label=g) for g in sorted(genes)]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=6, ncol=2)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "figures", "ontarget_rates.png"), dpi=200)
    plt.close(fig)


def plot_3prime_match_distribution(comp_df, random_df, output_dir):
    """Histogram of 3' consecutive match lengths: off-target vs random."""
    fig, ax = plt.subplots(figsize=(8, 5))

    max_val = max(
        comp_df["match_3prime_consecutive"].max(),
        random_df["match_3prime_consecutive"].max() if len(random_df) > 0 else 0,
    )
    bins = np.arange(-0.5, max_val + 1.5, 1)

    ax.hist(comp_df["match_3prime_consecutive"], bins=bins, density=True,
            alpha=0.6, label="Off-target sites", color="coral")
    if len(random_df) > 0:
        ax.hist(random_df["match_3prime_consecutive"], bins=bins, density=True,
                alpha=0.6, label="Random sites", color="steelblue")

    ax.set_xlabel("3' consecutive match length (bp)")
    ax.set_ylabel("Density")
    ax.set_title("3' end complementarity: off-target vs random")
    ax.legend()

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "figures", "3prime_match_distribution.png"), dpi=200)
    plt.close(fig)


def plot_primer_feature_correlations(summary, features, output_dir):
    """Scatter plots of primer features vs off-target rate."""
    merged = summary.merge(features, left_on="rt_primer", right_on="probe_name")

    feature_cols = [
        ("primer_length", "Primer length (bp)"),
        ("primer_gc", "GC content"),
        ("gc_3prime_8", "3' 8bp GC content"),
        ("max_homopolymer", "Max homopolymer run"),
        ("dinuc_entropy", "Dinucleotide entropy"),
        ("dG37", "dG at 37C (kcal/mol)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    for idx, (col, label) in enumerate(feature_cols):
        ax = axes[idx]
        x = merged[col]
        y = 1 - merged["on_target_rate"]  # off-target rate

        ax.scatter(x, y, s=30, alpha=0.7, edgecolors="black", linewidth=0.3)
        ax.set_xlabel(label)
        ax.set_ylabel("Off-target rate")

        # Add correlation
        r, pval = stats.spearmanr(x, y)
        ax.set_title(f"Spearman r={r:.2f}, p={pval:.2e}", fontsize=9)

        # Add trend line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_sorted = np.sort(x)
        ax.plot(x_sorted, p(x_sorted), "r--", alpha=0.5)

    plt.suptitle("Primer features vs off-target rate", fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "figures", "feature_correlations.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_reads_vs_complementarity(comp_df, output_dir):
    """Scatter: 3' match length vs number of reads at off-target sites."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(comp_df["match_3prime_consecutive"],
               comp_df["n_reads"],
               s=5, alpha=0.3, color="coral")
    ax.set_xlabel("3' consecutive match length (bp)")
    ax.set_ylabel("Number of reads at site")
    ax.set_yscale("log")
    ax.set_title("Off-target priming intensity vs 3' complementarity")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "figures", "reads_vs_complementarity.png"), dpi=200)
    plt.close(fig)


def plot_chromosome_distribution(comp_df, output_dir):
    """Bar plot of off-target site distribution across chromosomes."""
    chrom_order = [f"chr{i}" for i in range(1, 20)] + ["chrX", "chrY", "chrM"]
    chrom_counts = comp_df["chrom"].value_counts()
    chrom_counts = chrom_counts.reindex(chrom_order).dropna().astype(int)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(len(chrom_counts)), chrom_counts.values, color="steelblue",
           edgecolor="black", linewidth=0.3)
    ax.set_xticks(range(len(chrom_counts)))
    ax.set_xticklabels(chrom_counts.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Number of off-target priming sites")
    ax.set_title("Genome-wide distribution of off-target priming sites")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "figures", "chromosome_distribution.png"), dpi=200)
    plt.close(fig)


def plot_roc_curve(y_true, y_proba, model_name, output_dir):
    """Plot ROC curve for a classifier."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="darkorange", lw=2,
            label=f"{model_name} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC: Predicting off-target priming sites")
    ax.legend(loc="lower right")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "figures", f"roc_{model_name.lower().replace(' ', '_')}.png"),
                dpi=200)
    plt.close(fig)
    return roc_auc


def plot_feature_importance(importances, feature_names, output_dir):
    """Plot feature importance from random forest."""
    sorted_idx = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(len(importances)), importances[sorted_idx],
            color="steelblue", edgecolor="black", linewidth=0.3)
    ax.set_yticks(range(len(importances)))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=9)
    ax.set_xlabel("Feature importance")
    ax.set_title("Random Forest: Feature importance for predicting off-target priming")
    ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "figures", "feature_importance.png"), dpi=200)
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    os.makedirs(os.path.join(args.output_dir, "figures"), exist_ok=True)

    # Load data
    print("Loading data...", file=sys.stderr)
    comp_df = pd.read_csv(args.complementarity, sep="\t")
    summary = pd.read_csv(args.primer_summary, sep="\t")
    primers = pd.read_csv(args.primers)
    primer_seqs = dict(zip(primers["ProbeName"], primers["Seq"]))

    print(f"  Off-target sites: {len(comp_df):,}", file=sys.stderr)
    print(f"  Primers: {len(summary)}", file=sys.stderr)

    # --------------------------------------------------------
    # 1. Compute primer features
    # --------------------------------------------------------
    print("\n1. Computing primer features...", file=sys.stderr)
    features = compute_primer_features(primers)

    # --------------------------------------------------------
    # 2. Generate random control sequences
    # --------------------------------------------------------
    print("\n2. Generating random control complementarity...", file=sys.stderr)
    if args.random_controls and os.path.exists(args.random_controls):
        random_df = pd.read_csv(args.random_controls, sep="\t")
        print(f"  Loaded {len(random_df):,} pre-computed random controls", file=sys.stderr)
    else:
        random_seqs = generate_random_sequences(args.n_random_sites, 40)
        rng = np.random.default_rng(42)
        primers_list = list(primer_seqs.items())
        random_records = []
        for seq in random_seqs:
            pname, pseq = primers_list[rng.integers(len(primers_list))]
            metrics = _analyze_site_inline(pseq, seq)
            metrics["rt_primer"] = pname
            random_records.append(metrics)
        random_df = pd.DataFrame(random_records)
        print(f"  Generated {len(random_df):,} random controls", file=sys.stderr)

    # --------------------------------------------------------
    # 3. Visualizations
    # --------------------------------------------------------
    print("\n3. Generating visualizations...", file=sys.stderr)

    plot_ontarget_rates(summary, args.output_dir)
    print("  - On-target rates bar chart", file=sys.stderr)

    plot_3prime_match_distribution(comp_df, random_df, args.output_dir)
    print("  - 3' match distribution histogram", file=sys.stderr)

    plot_primer_feature_correlations(summary, features, args.output_dir)
    print("  - Primer feature correlation plots", file=sys.stderr)

    plot_reads_vs_complementarity(comp_df, args.output_dir)
    print("  - Reads vs complementarity scatter", file=sys.stderr)

    plot_chromosome_distribution(comp_df, args.output_dir)
    print("  - Chromosome distribution", file=sys.stderr)

    # --------------------------------------------------------
    # 4. Statistical tests: off-target vs random
    # --------------------------------------------------------
    print("\n4. Statistical comparison: off-target vs random...", file=sys.stderr)
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("RT Primer Non-Specific Priming: Rule Inference Report")
    report_lines.append("=" * 70)

    for metric in ["match_3prime_consecutive", "matches_last_8", "best_contiguous_match"]:
        if metric not in random_df.columns:
            continue
        offtarget_vals = comp_df[metric].values
        random_vals = random_df[metric].values
        stat, pval = stats.mannwhitneyu(offtarget_vals, random_vals, alternative="greater")
        effect_size = offtarget_vals.mean() - random_vals.mean()

        report_lines.append(f"\n{metric}:")
        report_lines.append(f"  Off-target mean: {offtarget_vals.mean():.2f}")
        report_lines.append(f"  Random mean:     {random_vals.mean():.2f}")
        report_lines.append(f"  Difference:      {effect_size:.2f}")
        report_lines.append(f"  Mann-Whitney U:  {stat:.0f}")
        report_lines.append(f"  p-value:         {pval:.2e}")

    # --------------------------------------------------------
    # 5. Predictive modeling: off-target sites vs random
    # --------------------------------------------------------
    print("\n5. Building predictive models...", file=sys.stderr)

    feature_cols = ["match_3prime_consecutive", "matches_last_6", "matches_last_8",
                    "matches_last_10", "matches_last_12"]
    available_cols = [c for c in feature_cols if c in comp_df.columns and c in random_df.columns]

    # Prepare labeled dataset
    offtarget_features = comp_df[available_cols].copy()
    offtarget_features["label"] = 1

    random_features = random_df[available_cols].copy()
    random_features["label"] = 0

    # Balance classes by downsampling the larger class
    n_min = min(len(offtarget_features), len(random_features))
    offtarget_sample = offtarget_features.sample(n=n_min, random_state=42)
    random_sample = random_features.sample(n=n_min, random_state=42)

    combined = pd.concat([offtarget_sample, random_sample], ignore_index=True)
    X = combined[available_cols].values
    y = combined["label"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Logistic Regression with cross-validation
    lr = LogisticRegression(random_state=42, max_iter=1000)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_proba_lr = cross_val_predict(lr, X_scaled, y, cv=cv, method="predict_proba")[:, 1]
    lr_auc = plot_roc_curve(y, y_proba_lr, "Logistic Regression", args.output_dir)

    # Fit final LR model for coefficients
    lr.fit(X_scaled, y)
    report_lines.append(f"\n{'='*70}")
    report_lines.append("Logistic Regression Coefficients:")
    for col, coef in zip(available_cols, lr.coef_[0]):
        report_lines.append(f"  {col:35s}: {coef:+.4f}")
    report_lines.append(f"  AUC (5-fold CV): {lr_auc:.3f}")

    # Random Forest with cross-validation
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    y_proba_rf = cross_val_predict(rf, X, y, cv=cv, method="predict_proba")[:, 1]
    rf_auc = plot_roc_curve(y, y_proba_rf, "Random Forest", args.output_dir)

    # Fit final RF for feature importance
    rf.fit(X, y)
    plot_feature_importance(rf.feature_importances_, available_cols, args.output_dir)

    report_lines.append(f"\nRandom Forest Feature Importance:")
    for col, imp in sorted(zip(available_cols, rf.feature_importances_),
                           key=lambda x: -x[1]):
        report_lines.append(f"  {col:35s}: {imp:.4f}")
    report_lines.append(f"  AUC (5-fold CV): {rf_auc:.3f}")

    # --------------------------------------------------------
    # 6. Per-primer off-target rate analysis
    # --------------------------------------------------------
    print("\n6. Per-primer feature analysis...", file=sys.stderr)
    merged = summary.merge(features, left_on="rt_primer", right_on="probe_name")
    offtarget_rate = 1 - merged["on_target_rate"]

    report_lines.append(f"\n{'='*70}")
    report_lines.append("Primer Feature Correlations with Off-Target Rate:")
    for col in ["primer_length", "primer_gc", "gc_3prime_8", "max_homopolymer",
                "dinuc_entropy", "dG37"]:
        r, pval = stats.spearmanr(merged[col], offtarget_rate)
        report_lines.append(f"  {col:25s}: Spearman r={r:+.3f}, p={pval:.3e}")

    # 3' terminal base effect
    report_lines.append(f"\nOff-target rate by 3' terminal base:")
    for base in ["A", "T", "G", "C"]:
        mask = merged["terminal_base"] == base
        if mask.sum() > 0:
            mean_rate = offtarget_rate[mask].mean()
            report_lines.append(f"  {base}: {mean_rate:.3f} (n={mask.sum()})")

    # --------------------------------------------------------
    # 7. Minimum complementarity for priming
    # --------------------------------------------------------
    report_lines.append(f"\n{'='*70}")
    report_lines.append("Minimum Complementarity Analysis:")

    # Weighted by read count
    weighted_match = np.average(
        comp_df["match_3prime_consecutive"],
        weights=comp_df["n_reads"],
    )
    report_lines.append(f"  Read-weighted mean 3' consecutive match: {weighted_match:.1f} bp")

    # Sites with high read count (strong off-target priming)
    top_sites = comp_df.nlargest(100, "n_reads")
    report_lines.append(f"\n  Top 100 off-target sites by read count:")
    report_lines.append(f"    Mean 3' consecutive match: {top_sites['match_3prime_consecutive'].mean():.1f} bp")
    report_lines.append(f"    Mean matches in last 8bp:  {top_sites['matches_last_8'].mean():.1f}/8")
    report_lines.append(f"    Min 3' consecutive match:  {top_sites['match_3prime_consecutive'].min()} bp")

    # --------------------------------------------------------
    # Write report
    # --------------------------------------------------------
    report_path = os.path.join(args.output_dir, "model_summary.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
        f.write("\n")

    print(f"\nReport written to: {report_path}", file=sys.stderr)
    print(f"Figures written to: {os.path.join(args.output_dir, 'figures')}/", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
