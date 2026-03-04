"""
Analyze sequence complementarity between RT primers and off-target priming sites.

For each off-target site, we compare the RT primer sequence to the genomic sequence
at the priming site. The 3' end of the primer is most critical for initiating
reverse transcription, so we focus on 3' end complementarity metrics.

The genomic_seq from step 03 is already oriented so that a perfect on-target match
would align directly with the primer's Seq (the probe sequence).

Usage:
    python 04_analyze_complementarity.py \
        --sites offtarget_sites.tsv.gz \
        --primers ../data/rt_primers.csv \
        --output complementarity_analysis.tsv.gz
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import helpers


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Analyze primer-site sequence complementarity at off-target sites.",
    )
    parser.add_argument(
        "--sites", required=True,
        help="Path to offtarget_sites.tsv.gz from step 03.",
    )
    parser.add_argument(
        "--primers", required=True,
        help="Path to RT primer CSV file.",
    )
    parser.add_argument(
        "--output", required=True,
        help="Path to output complementarity_analysis.tsv.gz.",
    )
    parser.add_argument(
        "--exclude-bed",
        help="BED file of regions to exclude (e.g., rRNA loci). "
             "Any off-target site overlapping these regions is filtered out.",
    )
    return parser.parse_args()


# ============================================================
# Complementarity analysis functions
# ============================================================

def count_3prime_consecutive_match(primer_seq, site_seq):
    """
    Count consecutive matching bases from the 3' end of the primer.
    The site_seq should be oriented so that direct comparison to primer_seq
    is meaningful (i.e., both in the same 5'->3' direction).
    """
    count = 0
    # Align from the 3' end of the primer against the 3' end of the site sequence
    for p, s in zip(reversed(primer_seq), reversed(site_seq)):
        if p.upper() == s.upper():
            count += 1
        else:
            break
    return count


def count_matches_in_window(primer_seq, site_seq, window_start_from_3prime, window_size):
    """
    Count matching bases in a window of the primer, counted from the 3' end.
    window_start_from_3prime=0, window_size=8 means the last 8 bases.
    """
    p_end = len(primer_seq)
    p_start = max(0, p_end - window_start_from_3prime - window_size)
    p_stop = p_end - window_start_from_3prime

    s_end = len(site_seq)
    s_start = max(0, s_end - window_start_from_3prime - window_size)
    s_stop = s_end - window_start_from_3prime

    if p_start >= p_stop or s_start >= s_stop:
        return 0, 0

    primer_window = primer_seq[p_start:p_stop].upper()
    site_window = site_seq[s_start:s_stop].upper()

    min_len = min(len(primer_window), len(site_window))
    # Align from the right (3' end)
    primer_aligned = primer_window[-min_len:]
    site_aligned = site_window[-min_len:]

    matches = sum(1 for p, s in zip(primer_aligned, site_aligned) if p == s)
    return matches, min_len


def count_wobble_pairs(primer_seq, site_seq):
    """
    Count G:T (wobble) base pairs between the primer and site.
    In DNA terms, G in one strand can pair with T in the other (weaker than Watson-Crick
    but thermodynamically tolerated).
    """
    wobble_pairs = {("G", "T"), ("T", "G")}
    count = 0
    min_len = min(len(primer_seq), len(site_seq))
    # Align from the 3' end
    for p, s in zip(primer_seq[-min_len:], site_seq[-min_len:]):
        if (p.upper(), s.upper()) in wobble_pairs:
            count += 1
    return count


def best_local_match(primer_seq, site_seq):
    """
    Find the best contiguous match between the primer and site sequence
    by sliding the primer across the site. Returns (best_match_length, best_offset).
    """
    primer_seq = primer_seq.upper()
    site_seq = site_seq.upper()

    best_match = 0
    best_offset = 0

    # Slide primer across site sequence
    for offset in range(-(len(primer_seq) - 1), len(site_seq)):
        current_match = 0
        max_match_at_offset = 0

        for i in range(len(primer_seq)):
            j = i + offset
            if 0 <= j < len(site_seq):
                if primer_seq[i] == site_seq[j]:
                    current_match += 1
                    max_match_at_offset = max(max_match_at_offset, current_match)
                else:
                    current_match = 0

        if max_match_at_offset > best_match:
            best_match = max_match_at_offset
            best_offset = offset

    return best_match, best_offset


def total_matches_aligned_3prime(primer_seq, site_seq):
    """
    Count total matching bases when the 3' ends of primer and site are aligned.
    """
    primer_seq = primer_seq.upper()
    site_seq = site_seq.upper()
    min_len = min(len(primer_seq), len(site_seq))
    matches = sum(
        1 for p, s in zip(primer_seq[-min_len:], site_seq[-min_len:])
        if p == s
    )
    return matches, min_len


def analyze_site(primer_seq, site_seq):
    """Compute all complementarity metrics for one primer-site pair."""
    if not site_seq or "N" * 5 in site_seq.upper():
        return {
            "match_3prime_consecutive": 0,
            "matches_last_6": 0, "window_last_6": 0,
            "matches_last_8": 0, "window_last_8": 0,
            "matches_last_10": 0, "window_last_10": 0,
            "matches_last_12": 0, "window_last_12": 0,
            "total_matches_aligned": 0, "alignment_length": 0,
            "best_contiguous_match": 0,
            "wobble_pairs": 0,
            "site_gc": 0.0,
        }

    match_3p = count_3prime_consecutive_match(primer_seq, site_seq)

    m6, w6 = count_matches_in_window(primer_seq, site_seq, 0, 6)
    m8, w8 = count_matches_in_window(primer_seq, site_seq, 0, 8)
    m10, w10 = count_matches_in_window(primer_seq, site_seq, 0, 10)
    m12, w12 = count_matches_in_window(primer_seq, site_seq, 0, 12)

    total_m, align_len = total_matches_aligned_3prime(primer_seq, site_seq)
    best_contig, _ = best_local_match(primer_seq, site_seq)
    wobble = count_wobble_pairs(primer_seq, site_seq)

    gc_count = sum(1 for b in site_seq.upper() if b in "GC")
    site_gc = gc_count / len(site_seq) if site_seq else 0.0

    return {
        "match_3prime_consecutive": match_3p,
        "matches_last_6": m6, "window_last_6": w6,
        "matches_last_8": m8, "window_last_8": w8,
        "matches_last_10": m10, "window_last_10": w10,
        "matches_last_12": m12, "window_last_12": w12,
        "total_matches_aligned": total_m, "alignment_length": align_len,
        "best_contiguous_match": best_contig,
        "wobble_pairs": wobble,
        "site_gc": round(site_gc, 3),
    }


def load_exclude_regions(bed_path):
    """Load a BED file of regions to exclude. Returns dict of chrom -> list of (start, end)."""
    regions = {}
    with open(bed_path) as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            parts = line.strip().split("\t")
            chrom = parts[0]
            start = int(parts[1])
            end = int(parts[2])
            regions.setdefault(chrom, []).append((start, end))
    # Sort intervals for each chromosome
    for chrom in regions:
        regions[chrom].sort()
    return regions


def site_in_excluded_region(chrom, pos, exclude_regions):
    """Check if a position falls within any excluded region."""
    if chrom not in exclude_regions:
        return False
    for start, end in exclude_regions[chrom]:
        if start <= pos < end:
            return True
        if pos < start:
            break  # sorted, no need to check further
    return False


def main():
    args = parse_args()

    print("Loading primers...", file=sys.stderr)
    primers = pd.read_csv(args.primers)
    primer_seqs = dict(zip(primers["ProbeName"], primers["Seq"]))

    print("Loading off-target sites...", file=sys.stderr)
    sites = pd.read_csv(args.sites, sep="\t")
    print(f"  Loaded {len(sites):,} off-target priming sites", file=sys.stderr)

    # Filter out excluded regions (e.g., rRNA loci)
    if args.exclude_bed:
        exclude_regions = load_exclude_regions(args.exclude_bed)
        n_before = len(sites)
        mask = sites.apply(
            lambda row: not site_in_excluded_region(row["chrom"], row["site_pos"], exclude_regions),
            axis=1,
        )
        sites = sites[mask].copy()
        n_excluded = n_before - len(sites)
        print(f"  Excluded {n_excluded:,} sites in BED regions, "
              f"{len(sites):,} remaining", file=sys.stderr)

    # Analyze complementarity for each site
    print("Analyzing complementarity...", file=sys.stderr)
    results = []
    for _, row in tqdm(sites.iterrows(), total=len(sites)):
        rt_primer = row["rt_primer"]
        site_seq = row["genomic_seq"]
        primer_seq = primer_seqs.get(rt_primer, "")

        if not primer_seq:
            continue

        metrics = analyze_site(primer_seq, site_seq)
        metrics["rt_primer"] = rt_primer
        metrics["chrom"] = row["chrom"]
        metrics["site_pos"] = row["site_pos"]
        metrics["strand"] = row["strand"]
        metrics["n_reads"] = row["n_reads"]
        metrics["genomic_seq"] = site_seq
        metrics["primer_seq"] = primer_seq
        results.append(metrics)

    result_df = pd.DataFrame(results)

    # Reorder columns
    col_order = [
        "rt_primer", "chrom", "site_pos", "strand", "n_reads",
        "primer_seq", "genomic_seq",
        "match_3prime_consecutive",
        "matches_last_6", "window_last_6",
        "matches_last_8", "window_last_8",
        "matches_last_10", "window_last_10",
        "matches_last_12", "window_last_12",
        "total_matches_aligned", "alignment_length",
        "best_contiguous_match", "wobble_pairs", "site_gc",
    ]
    result_df = result_df[col_order]

    # Write output
    print(f"Writing {len(result_df):,} analyzed sites to {args.output}...", file=sys.stderr)
    result_df.to_csv(args.output, sep="\t", index=False, compression="gzip")

    # Summary statistics
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Complementarity Analysis Summary", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"Sites analyzed: {len(result_df):,}", file=sys.stderr)
    print(f"\n3' consecutive match distribution:", file=sys.stderr)
    match_dist = result_df["match_3prime_consecutive"].value_counts().sort_index()
    for length, count in match_dist.items():
        pct = count / len(result_df) * 100
        print(f"  {length:>3d} bp: {count:>8,} sites ({pct:5.1f}%)", file=sys.stderr)

    print(f"\nMatches in last 8 bases of primer:", file=sys.stderr)
    for n in range(9):
        count = (result_df["matches_last_8"] == n).sum()
        pct = count / len(result_df) * 100
        print(f"  {n}/8: {count:>8,} sites ({pct:5.1f}%)", file=sys.stderr)

    print(f"\nBest contiguous match (anywhere):", file=sys.stderr)
    print(f"  Mean:   {result_df['best_contiguous_match'].mean():.1f} bp", file=sys.stderr)
    print(f"  Median: {result_df['best_contiguous_match'].median():.0f} bp", file=sys.stderr)
    print(f"  Max:    {result_df['best_contiguous_match'].max():.0f} bp", file=sys.stderr)


if __name__ == "__main__":
    main()
