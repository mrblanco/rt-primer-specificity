"""
Classify reads as on-target or off-target for each RT primer.

On-target: read aligns to the same chromosome as the primer's target AND the read's
5' alignment position is within 100kb of the primer's target region.

Usage:
    python 02_classify_ontarget.py \
        --reads read_info.tsv.gz \
        --primers ../data/rt_primers.csv \
        --output classified_reads.tsv.gz \
        --summary primer_summary.tsv
"""

import argparse
import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import helpers

# Maximum distance (bp) from primer target region to count as on-target
ON_TARGET_WINDOW = 100_000


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Classify reads as on-target or off-target per RT primer.",
    )
    parser.add_argument(
        "--reads", required=True,
        help="Path to read_info.tsv.gz from step 01.",
    )
    parser.add_argument(
        "--primers", required=True,
        help="Path to RT primer CSV file.",
    )
    parser.add_argument(
        "--output", required=True,
        help="Path to output classified_reads.tsv.gz.",
    )
    parser.add_argument(
        "--summary", required=True,
        help="Path to output primer_summary.tsv.",
    )
    parser.add_argument(
        "--window", type=int, default=ON_TARGET_WINDOW,
        help="Distance window (bp) from target region for on-target classification.",
    )
    return parser.parse_args()


def load_primer_targets(primer_csv: str) -> pd.DataFrame:
    """Load primer target positions from the CSV."""
    primers = pd.read_csv(primer_csv)
    targets = primers[["ProbeName", "GeneName", "Chromosome", "theStartPos", "theEndPos",
                        "Target_Strand", "Seq", "GCpc", "dG37"]].copy()
    targets = targets.rename(columns={
        "ProbeName": "probe_name",
        "GeneName": "gene_name",
        "Chromosome": "target_chrom_num",
        "theStartPos": "target_start",
        "theEndPos": "target_end",
        "Target_Strand": "target_strand",
        "Seq": "primer_seq",
        "GCpc": "primer_gc",
        "dG37": "primer_dg37",
    })

    # Convert chromosome number to "chrN" format
    targets["target_chrom"] = targets["target_chrom_num"].apply(
        lambda x: f"chr{x}"
    )

    targets["primer_length"] = targets["primer_seq"].str.len()

    return targets


def classify_read(row, targets_dict, window):
    """Classify a single read as on-target or off-target."""
    rt_primer = row["rt_primer"]
    if rt_primer not in targets_dict:
        return False, -1

    target = targets_dict[rt_primer]
    target_chrom = target["target_chrom"]
    target_start = target["target_start"]
    target_end = target["target_end"]

    # Check chromosome match
    if row["chrom"] != target_chrom:
        return False, -1

    # Compute distance from read position to target region
    pos = row["pos"]
    if pos < target_start:
        distance = target_start - pos
    elif pos > target_end:
        distance = pos - target_end
    else:
        distance = 0

    is_ontarget = distance <= window
    return is_ontarget, distance


def main():
    args = parse_args()

    print("Loading primer targets...", file=sys.stderr)
    targets = load_primer_targets(args.primers)
    targets_dict = {row["probe_name"]: row for _, row in targets.iterrows()}

    print("Loading reads...", file=sys.stderr)
    reads = pd.read_csv(args.reads, sep="\t")
    print(f"  Loaded {len(reads):,} reads", file=sys.stderr)

    # Classify each read
    print("Classifying reads...", file=sys.stderr)
    results = reads["rt_primer"].map(lambda x: x in targets_dict)
    known_primer_mask = results

    # Vectorized classification for speed
    reads["target_chrom"] = reads["rt_primer"].map(
        lambda x: targets_dict[x]["target_chrom"] if x in targets_dict else ""
    )
    reads["target_start"] = reads["rt_primer"].map(
        lambda x: targets_dict[x]["target_start"] if x in targets_dict else -1
    )
    reads["target_end"] = reads["rt_primer"].map(
        lambda x: targets_dict[x]["target_end"] if x in targets_dict else -1
    )

    # Check chromosome match
    chrom_match = reads["chrom"] == reads["target_chrom"]

    # Compute distance to target region
    reads["distance_to_target"] = 0
    mask_before = reads["pos"] < reads["target_start"]
    mask_after = reads["pos"] > reads["target_end"]
    reads.loc[mask_before, "distance_to_target"] = (
        reads.loc[mask_before, "target_start"] - reads.loc[mask_before, "pos"]
    )
    reads.loc[mask_after, "distance_to_target"] = (
        reads.loc[mask_after, "pos"] - reads.loc[mask_after, "target_end"]
    )

    # Reads on wrong chromosome get distance = -1
    reads.loc[~chrom_match, "distance_to_target"] = -1

    # On-target = same chromosome AND within window
    reads["is_ontarget"] = chrom_match & (reads["distance_to_target"] <= args.window)
    reads.loc[~chrom_match, "is_ontarget"] = False

    # Clean up temporary columns
    reads = reads.drop(columns=["target_chrom", "target_start", "target_end"])

    # Write classified reads
    print(f"Writing classified reads to {args.output}...", file=sys.stderr)
    reads.to_csv(args.output, sep="\t", index=False, compression="gzip")

    # Generate per-primer summary
    print("Generating primer summary...", file=sys.stderr)
    summary = reads.groupby("rt_primer").agg(
        total_reads=("is_ontarget", "size"),
        on_target=("is_ontarget", "sum"),
    ).reset_index()
    summary["off_target"] = summary["total_reads"] - summary["on_target"]
    summary["on_target_rate"] = summary["on_target"] / summary["total_reads"]

    # Add primer metadata
    primer_meta = targets[["probe_name", "gene_name", "primer_seq", "primer_length",
                           "primer_gc", "primer_dg37"]].copy()
    summary = summary.merge(primer_meta, left_on="rt_primer", right_on="probe_name", how="left")
    summary = summary.drop(columns=["probe_name"])
    summary = summary.sort_values("on_target_rate", ascending=True)

    summary.to_csv(args.summary, sep="\t", index=False)

    # Print summary statistics
    total = len(reads)
    on_target = reads["is_ontarget"].sum()
    off_target = total - on_target
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Classification Summary (window = {args.window:,} bp)", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"Total reads:    {total:>12,}", file=sys.stderr)
    print(f"On-target:      {on_target:>12,} ({on_target/total:.1%})", file=sys.stderr)
    print(f"Off-target:     {off_target:>12,} ({off_target/total:.1%})", file=sys.stderr)
    print(f"\nPer-primer summary written to: {args.summary}", file=sys.stderr)

    # Print top/bottom primers by on-target rate
    print(f"\nTop 5 primers by on-target rate:", file=sys.stderr)
    for _, row in summary.tail(5).iterrows():
        print(f"  {row['rt_primer']:30s}  {row['on_target_rate']:.1%}  "
              f"(n={row['total_reads']:,})", file=sys.stderr)

    print(f"\nBottom 5 primers by on-target rate:", file=sys.stderr)
    for _, row in summary.head(5).iterrows():
        print(f"  {row['rt_primer']:30s}  {row['on_target_rate']:.1%}  "
              f"(n={row['total_reads']:,})", file=sys.stderr)


if __name__ == "__main__":
    main()
