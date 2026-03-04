#!/bin/bash
#SBATCH --job-name=rt_seqextract
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --partition=short
#SBATCH --output=rt_seqextract_%j.out
#SBATCH --error=rt_seqextract_%j.err

# ============================================================
# Step 03: Extract genomic sequences at off-target priming sites
# Step 04: Analyze complementarity
# ============================================================

set -euo pipefail

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate rtprimer

# Paths — adjust as needed
GENOME="/groups/guttman/genomes/mm10/fasta/mm10.fa"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${SCRIPT_DIR}/../data"
RESULTS_DIR="${SCRIPT_DIR}/../results"

echo "=== Step 03: Extracting genomic sequences at off-target sites ==="
python "${SCRIPT_DIR}/03_extract_offtarget_seqs.py" \
    --reads "${RESULTS_DIR}/classified_reads.tsv.gz" \
    --genome "${GENOME}" \
    --output "${RESULTS_DIR}/offtarget_sites.tsv.gz" \
    --progress

echo ""
echo "=== Step 04: Analyzing complementarity ==="
python "${SCRIPT_DIR}/04_analyze_complementarity.py" \
    --sites "${RESULTS_DIR}/offtarget_sites.tsv.gz" \
    --primers "${DATA_DIR}/rt_primers.csv" \
    --output "${RESULTS_DIR}/complementarity_analysis.tsv.gz"

echo ""
echo "Done. Results in ${RESULTS_DIR}/"
