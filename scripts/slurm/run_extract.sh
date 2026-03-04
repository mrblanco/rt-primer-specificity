#!/bin/bash
#SBATCH --job-name=rt_extract
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --partition=short
#SBATCH --output=rt_extract_%j.out
#SBATCH --error=rt_extract_%j.err

# ============================================================
# Step 01: Extract read info from BAM
# Step 02: Classify on/off-target
# ============================================================

set -euo pipefail

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate rtprimer

# Paths — adjust as needed
BAM="/resnick/scratch/mblanco/pipeline/swift-seq/out_dir/workup/fastqs/100uM/raw/mm10SE/aligned.mouse.sorted.bam"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${SCRIPT_DIR}/../data"
RESULTS_DIR="${SCRIPT_DIR}/../results"

mkdir -p "${RESULTS_DIR}"

echo "=== Step 01: Extracting read info from BAM ==="
python "${SCRIPT_DIR}/01_extract_read_info.py" \
    -i "${BAM}" \
    -o "${RESULTS_DIR}/read_info.tsv.gz" \
    --progress

echo ""
echo "=== Step 02: Classifying on/off-target ==="
python "${SCRIPT_DIR}/02_classify_ontarget.py" \
    --reads "${RESULTS_DIR}/read_info.tsv.gz" \
    --primers "${DATA_DIR}/rt_primers.csv" \
    --output "${RESULTS_DIR}/classified_reads.tsv.gz" \
    --summary "${RESULTS_DIR}/primer_summary.tsv"

echo ""
echo "Done. Results in ${RESULTS_DIR}/"
