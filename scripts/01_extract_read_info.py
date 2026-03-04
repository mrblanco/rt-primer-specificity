"""
Extract read information from a BAM file, parsing RT primer identity from read names.

Read names are formatted by cutadapt as: {original_id}-{r1_adapter}-{rt_primer_name}
where rt_primer_name is the probe name (e.g., Atrx_probe_76) or "no_adapter".

Only mapped reads with an identified RT primer are output.

Usage:
    python 01_extract_read_info.py -i aligned.mouse.sorted.bam -o read_info.tsv.gz
"""

import argparse
import gzip
import os
import sys

import pysam
from tqdm.auto import tqdm

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import helpers


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Extract read info and RT primer identity from a BAM file.",
    )
    parser.add_argument(
        "-i", "--input", required=True, metavar="in.bam",
        help="Path to input BAM file (coordinate-sorted, indexed).",
    )
    parser.add_argument(
        "-o", "--output", required=True, metavar="out.tsv.gz",
        help="Path to output TSV file (gzip compressed).",
    )
    parser.add_argument(
        "--min-mapq", type=int, default=0,
        help="Minimum mapping quality to include a read.",
    )
    parser.add_argument(
        "--progress", action="store_true",
        help="Display a progress bar.",
    )
    return parser.parse_args()


def extract_rt_primer(query_name: str) -> str:
    """
    Parse RT primer name from a cutadapt-renamed read name.
    Format: {original_id}-{r1_adapter}-{rt_primer_name}
    We split from the right on '-' to handle original IDs that may contain hyphens.

    The rt_primer_name always starts with a gene name followed by '_probe_',
    so we look for that pattern to find the correct split point.
    """
    # Look for the probe name pattern: GeneName_probe_NNN at the end
    # This is more robust than rsplit since original IDs and adapter names can contain hyphens
    idx = query_name.rfind("_probe_")
    if idx == -1:
        return "no_adapter"

    # Walk backwards from the _probe_ position to find the start of the probe name
    # The probe name is preceded by a '-' delimiter
    probe_start = query_name.rfind("-", 0, idx)
    if probe_start == -1:
        return "no_adapter"

    # Check if this is actually "no_adapter" at the end
    rt_primer = query_name[probe_start + 1:]

    # There may be a second adapter field between the r1 adapter and the probe name.
    # The format is: {id}-{r1_adapter}-{rt_primer}
    # But the rt_primer itself could also be preceded by an r1_adapter name.
    # Since probe names always match GeneName_probe_NNN, extract just that.
    return rt_primer


def main():
    args = parse_args()

    header = [
        "rt_primer", "chrom", "pos", "end_pos", "strand", "mapq",
        "read_length", "is_read2",
    ]

    n_total = 0
    n_written = 0
    n_unmapped = 0
    n_no_primer = 0
    n_low_mapq = 0

    with pysam.AlignmentFile(args.input, mode="rb") as bam_in, \
         gzip.open(args.output, "wt") as f_out:

        f_out.write("\t".join(header) + "\n")

        for read in tqdm(bam_in.fetch(until_eof=True), disable=not args.progress):
            n_total += 1

            if read.is_unmapped:
                n_unmapped += 1
                continue

            if read.mapping_quality < args.min_mapq:
                n_low_mapq += 1
                continue

            rt_primer = extract_rt_primer(read.query_name)
            if rt_primer == "no_adapter":
                n_no_primer += 1
                continue

            chrom = read.reference_name
            # 5' end of the alignment (where the cDNA starts = where RT primed)
            if read.is_reverse:
                pos = read.reference_end  # 5' end on minus strand
                strand = "-"
            else:
                pos = read.reference_start  # 0-based 5' end on plus strand
                strand = "+"

            end_pos = read.reference_end if strand == "+" else read.reference_start
            mapq = read.mapping_quality
            read_length = read.query_length or 0
            is_read2 = 1 if read.is_read2 else 0

            f_out.write(f"{rt_primer}\t{chrom}\t{pos}\t{end_pos}\t{strand}\t{mapq}\t{read_length}\t{is_read2}\n")
            n_written += 1

    print(f"Total reads in BAM: {n_total:,}", file=sys.stderr)
    print(f"  Unmapped: {n_unmapped:,}", file=sys.stderr)
    print(f"  Low MAPQ (< {args.min_mapq}): {n_low_mapq:,}", file=sys.stderr)
    print(f"  No RT primer: {n_no_primer:,}", file=sys.stderr)
    print(f"  Written: {n_written:,}", file=sys.stderr)


if __name__ == "__main__":
    main()
