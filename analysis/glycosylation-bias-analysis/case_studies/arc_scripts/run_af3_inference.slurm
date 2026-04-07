#!/bin/bash
#SBATCH --job-name=af3-infer
#SBATCH --partition=medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/data/chem-proteindesign/sjoh5764/af3_case_study/logs/%x-%j.out
#SBATCH --error=/data/chem-proteindesign/sjoh5764/af3_case_study/logs/%x-%j.err

set -euo pipefail

# ARC project layout for this repository.
PROJECT_ROOT="/data/chem-proteindesign/sjoh5764/af3_case_study"
CONTAINER_PATH="/path/to/alphafold3.sif"   # EDIT THIS: absolute path to your Apptainer image.
JSON_INPUT="${1:-}"                         # job name OR absolute/relative path to a *_data.json file.
MODEL_DIR="${PROJECT_ROOT}/models"
DB_DIR="${PROJECT_ROOT}/databases"
OUTPUT_DIR="${PROJECT_ROOT}/output"
JAX_CACHE_DIR="${PROJECT_ROOT}/jax_cache"
LOG_DIR="${PROJECT_ROOT}/logs"

# EDIT THIS: change to the actual AlphaFold 3 run script path inside the container.
AF3_ENTRYPOINT="python /PATH/INSIDE/CONTAINER/run_alphafold.py"

if [[ -z "${JSON_INPUT}" ]]; then
  echo "Usage: sbatch run_af3_inference.slurm <job_name_or_data_json_path>"
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

if [[ "${JSON_INPUT}" == *_data.json ]]; then
  if [[ "${JSON_INPUT}" = /* ]]; then
    DATA_JSON_PATH="${JSON_INPUT}"
  elif [[ -f "${JSON_INPUT}" ]]; then
    DATA_JSON_PATH="$(realpath "${JSON_INPUT}")"
  elif [[ -f "${OUTPUT_DIR}/${JSON_INPUT}" ]]; then
    DATA_JSON_PATH="${OUTPUT_DIR}/${JSON_INPUT}"
  else
    DATA_JSON_PATH="${OUTPUT_DIR}/${JSON_INPUT}"
  fi
  JOB_NAME="$(basename "${DATA_JSON_PATH}" _data.json)"
else
  JOB_NAME="${JSON_INPUT}"
  DATA_JSON_PATH="${OUTPUT_DIR}/${JOB_NAME}/${JOB_NAME}_data.json"
fi

if [[ ! -f "${DATA_JSON_PATH}" ]]; then
  echo "ERROR: Data JSON not found: ${DATA_JSON_PATH}"
  exit 1
fi

if [[ ! -d "${LOG_DIR}" ]]; then
  echo "ERROR: Log directory does not exist: ${LOG_DIR}"
  echo "Create it before submitting so Slurm can open the log files."
  exit 1
fi

module load apptainer

mkdir -p "${OUTPUT_DIR}" "${JAX_CACHE_DIR}"

echo "Project root: ${PROJECT_ROOT}"
echo "Container:    ${CONTAINER_PATH}"
echo "Job name:     ${JOB_NAME}"
echo "Data JSON:    ${DATA_JSON_PATH}"
echo "Model dir:    ${MODEL_DIR}"
echo "DB dir:       ${DB_DIR}"
echo "Output dir:   ${OUTPUT_DIR}"
echo "JAX cache:    ${JAX_CACHE_DIR}"
echo "Running inference only."

# The same output directory is reused after the data-pipeline stage, so
# --force_output_dir=true is intentional here.
apptainer exec \
  --nv \
  --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" \
  "${CONTAINER_PATH}" \
  bash -lc "
    set -euo pipefail
    export PYTHONUNBUFFERED=1
    export XLA_PYTHON_CLIENT_PREALLOCATE=true
    export XLA_CLIENT_MEM_FRACTION=0.95
    ${AF3_ENTRYPOINT} \
      --json_path='${DATA_JSON_PATH}' \
      --model_dir='${MODEL_DIR}' \
      --db_dir='${DB_DIR}' \
      --output_dir='${OUTPUT_DIR}' \
      --run_data_pipeline=false \
      --run_inference=true \
      --force_output_dir=true \
      --jax_compilation_cache_dir='${JAX_CACHE_DIR}'
  "
