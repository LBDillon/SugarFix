#!/bin/bash
#SBATCH --job-name=af3-data
#SBATCH --partition=medium
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/data/chem-proteindesign/sjoh5764/af3_case_study/logs/%x-%j.out
#SBATCH --error=/data/chem-proteindesign/sjoh5764/af3_case_study/logs/%x-%j.err

set -euo pipefail

# ARC project layout for this repository.
PROJECT_ROOT="/data/chem-proteindesign/sjoh5764/af3_case_study"
CONTAINER_PATH="/path/to/alphafold3.sif"   # EDIT THIS: absolute path to your Apptainer image.
JSON_INPUT="${1:-}"                         # filename under input/ or an absolute JSON path.
MODEL_DIR="${PROJECT_ROOT}/models"
DB_DIR="${PROJECT_ROOT}/databases"
OUTPUT_DIR="${PROJECT_ROOT}/output"
INPUT_DIR="${PROJECT_ROOT}/input"
LOG_DIR="${PROJECT_ROOT}/logs"

# EDIT THIS: change to the actual AlphaFold 3 run script path inside the container.
AF3_ENTRYPOINT="python /PATH/INSIDE/CONTAINER/run_alphafold.py"

# Optional tuning. AlphaFold 3's own defaults cap Jackhmmer CPU use around 8.
JACKHMMER_N_CPU=8

if [[ -z "${JSON_INPUT}" ]]; then
  echo "Usage: sbatch run_af3_data_pipeline.slurm <json_filename_or_absolute_path>"
  exit 1
fi

if [[ "${CONTAINER_PATH}" == "/path/to/alphafold3.sif" ]]; then
  echo "ERROR: Edit CONTAINER_PATH near the top of this script."
  exit 1
fi

if [[ "${AF3_ENTRYPOINT}" == "python /PATH/INSIDE/CONTAINER/run_alphafold.py" ]]; then
  echo "ERROR: Edit AF3_ENTRYPOINT near the top of this script."
  exit 1
fi

if [[ "${JSON_INPUT}" = /* ]]; then
  JSON_PATH="${JSON_INPUT}"
else
  JSON_PATH="${INPUT_DIR}/${JSON_INPUT}"
fi

if [[ ! -f "${JSON_PATH}" ]]; then
  echo "ERROR: Input JSON not found: ${JSON_PATH}"
  exit 1
fi

if [[ ! -d "${LOG_DIR}" ]]; then
  echo "ERROR: Log directory does not exist: ${LOG_DIR}"
  echo "Create it before submitting so Slurm can open the log files."
  exit 1
fi

module load apptainer

mkdir -p "${OUTPUT_DIR}"

echo "Project root: ${PROJECT_ROOT}"
echo "Container:    ${CONTAINER_PATH}"
echo "JSON input:   ${JSON_PATH}"
echo "Model dir:    ${MODEL_DIR}"
echo "DB dir:       ${DB_DIR}"
echo "Output dir:   ${OUTPUT_DIR}"
echo "Running data pipeline only."

apptainer exec \
  --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" \
  "${CONTAINER_PATH}" \
  bash -lc "
    set -euo pipefail
    export PYTHONUNBUFFERED=1
    ${AF3_ENTRYPOINT} \
      --json_path='${JSON_PATH}' \
      --model_dir='${MODEL_DIR}' \
      --db_dir='${DB_DIR}' \
      --output_dir='${OUTPUT_DIR}' \
      --run_data_pipeline=true \
      --run_inference=false \
      --jackhmmer_n_cpu='${JACKHMMER_N_CPU}'
  "
