"""
Extract genomic sequences at off-target priming sites from the mm10 genome FASTA.

For each off-target read, the priming site is where the RT primer hybridized to RNA.
The alignment start (5' end) of the read marks the 3' end of the primer binding site.
We extract 40bp of genomic sequence upstream of this position (the region where the
primer would have bound).

Off-target reads are clustered by position (within 5bp on same strand) to define
unique priming sites.

Usage:
    python 03_extract_offtarget_seqs.py \
        --reads classified_reads.tsv.gz \
        --genome /groups/guttman/genomes/mm10/fasta/mm10.fa \
        --output offtarget_sites.tsv.gz
"""

import argparse
import os
import sys
from collections import defaultdict

import pandas as pd
import pysam
from tqdm.auto import tqdm

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import helpers

# How many bp of genomic sequence to extract at each priming site
EXTRACT_LENGTH = 40

# Cluster reads within this distance into the same priming site
CLUSTER_DISTANCE = 5


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Extract genomic sequences at off-target priming sites.",
    )
    parser.add_argument(
        "--reads", required=True,
        help="Path to classified_reads.tsv.gz from step 02.",
    )
    parser.add_argument(
        "--genome", required=True,
        help="Path to mm10 genome FASTA (must be indexed with samtools faidx).",
    )
    parser.add_argument(
        "--output", required=True,
        help="Path to output offtarget_sites.tsv.gz.",
    )
    parser.add_argument(
        "--extract-length", type=int, default=EXTRACT_LENGTH,
        help="Length of genomic sequence to extract at each site.",
    )
    parser.add_argument(
        "--cluster-distance", type=int, default=CLUSTER_DISTANCE,
        help="Maximum distance to cluster reads into the same priming site.",
    )
    parser.add_argument(
        "--progress", action="store_true",
        help="Display a progress bar.",
    )
    return parser.parse_args()


def cluster_positions(positions, max_distance):
    """
    Cluster sorted positions that are within max_distance of each other.
    Returns list of (representative_pos, count) tuples.
    """
    if not positions:
        return []

    clusters = []
    cluster_start = positions[0]
    cluster_positions_list = [positions[0]]

    for pos in positions[1:]:
        if pos - cluster_start <= max_distance:
            cluster_positions_list.append(pos)
        else:
            # Use median position as representative
            rep_pos = cluster_positions_list[len(cluster_positions_list) // 2]
            clusters.append((rep_pos, len(cluster_positions_list)))
            cluster_start = pos
            cluster_positions_list = [pos]

    # Don't forget the last cluster
    rep_pos = cluster_positions_list[len(cluster_positions_list) // 2]
    clusters.append((rep_pos, len(cluster_positions_list)))

    return clusters


def extract_priming_site_seq(genome, chrom, pos, strand, extract_length):
    """
    Extract the genomic sequence at a priming site, oriented so that a perfect
    on-target match aligns directly with the primer Seq (5'→3').

    The R2 read's 5' alignment position (pos) marks the 3' end of the primer
    binding site. The primer extends UPSTREAM from pos.

    For a + strand R2 read:
        - cDNA is on + strand, RNA template was - strand-like
        - Primer Seq matches the + strand genomic DNA at the binding site
        - Extract + strand seq to the LEFT of pos → compare directly with Seq

    For a - strand R2 read:
        - cDNA is on - strand, RNA template was + strand-like
        - Primer Seq matches the - strand genomic DNA at the binding site
        - Extract + strand seq to the RIGHT of pos → reverse complement → compare with Seq
    """
    try:
        chrom_length = genome.get_reference_length(chrom)
    except ValueError:
        return None

    if strand == "+":
        # R2 on + strand: primer bound + strand DNA upstream (to the left)
        # + strand genomic seq here matches Seq directly
        start = max(0, pos - extract_length)
        end = pos
        seq = genome.fetch(chrom, start, end).upper()
    else:
        # R2 on - strand: primer bound - strand DNA downstream (to the right in + coords)
        # Extract + strand, then RC to get - strand = what matches Seq
        start = pos
        end = min(chrom_length, pos + extract_length)
        seq = genome.fetch(chrom, start, end).upper()
        seq = helpers.reverse_complement(seq)

    return seq


def main():
    args = parse_args()

    print("Loading classified reads...", file=sys.stderr)
    reads = pd.read_csv(args.reads, sep="\t")

    # Filter to off-target reads only
    off_target = reads[~reads["is_ontarget"]].copy()
    print(f"  Total reads: {len(reads):,}", file=sys.stderr)
    print(f"  Off-target reads: {len(off_target):,}", file=sys.stderr)

    # Group by primer, chrom, strand and cluster positions
    print("Clustering off-target positions into priming sites...", file=sys.stderr)
    site_records = []

    grouped = off_target.groupby(["rt_primer", "chrom", "strand"])
    for (rt_primer, chrom, strand), group in tqdm(grouped, disable=not args.progress):
        positions = sorted(group["pos"].tolist())
        clusters = cluster_positions(positions, args.cluster_distance)
        for rep_pos, n_reads in clusters:
            site_records.append({
                "rt_primer": rt_primer,
                "chrom": chrom,
                "site_pos": rep_pos,
                "strand": strand,
                "n_reads": n_reads,
            })

    sites = pd.DataFrame(site_records)
    print(f"  Unique priming sites: {len(sites):,}", file=sys.stderr)

    # Extract genomic sequences
    print("Extracting genomic sequences at priming sites...", file=sys.stderr)
    genome = pysam.FastaFile(args.genome)

    sequences = []
    for _, row in tqdm(sites.iterrows(), total=len(sites), disable=not args.progress):
        seq = extract_priming_site_seq(
            genome, row["chrom"], row["site_pos"], row["strand"], args.extract_length
        )
        sequences.append(seq if seq else "")

    genome.close()

    sites["genomic_seq"] = sequences

    # Filter out sites where sequence extraction failed
    n_before = len(sites)
    sites = sites[sites["genomic_seq"].str.len() > 0]
    n_after = len(sites)
    if n_before != n_after:
        print(f"  Dropped {n_before - n_after} sites with failed sequence extraction",
              file=sys.stderr)

    # Sort by number of reads (most-used priming sites first)
    sites = sites.sort_values("n_reads", ascending=False)

    # Write output
    print(f"Writing {len(sites):,} priming sites to {args.output}...", file=sys.stderr)
    sites.to_csv(args.output, sep="\t", index=False, compression="gzip")

    # Summary stats
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Off-target priming site summary", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"Unique sites:        {len(sites):>10,}", file=sys.stderr)
    print(f"Total off-target reads: {sites['n_reads'].sum():>10,}", file=sys.stderr)
    print(f"Median reads/site:   {sites['n_reads'].median():>10.0f}", file=sys.stderr)
    print(f"Max reads/site:      {sites['n_reads'].max():>10,}", file=sys.stderr)

    # Per-primer site counts
    primer_sites = sites.groupby("rt_primer").agg(
        n_sites=("site_pos", "count"),
        total_reads=("n_reads", "sum"),
    ).sort_values("n_sites", ascending=False)
    print(f"\nTop 10 primers by number of off-target sites:", file=sys.stderr)
    for name, row in primer_sites.head(10).iterrows():
        print(f"  {name:30s}  sites={row['n_sites']:>6,}  reads={row['total_reads']:>8,}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
