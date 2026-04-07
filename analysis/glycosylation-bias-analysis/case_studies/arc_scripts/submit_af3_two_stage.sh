#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/data/chem-proteindesign/sjoh5764/af3_case_study"
ARC_SCRIPTS_DIR="${PROJECT_ROOT}/code/arc_scripts"
INPUT_DIR="${PROJECT_ROOT}/input"
OUTPUT_DIR="${PROJECT_ROOT}/output"

JSON_INPUT="${1:-}"
JOB_NAME="${2:-}"

if [[ -z "${JSON_INPUT}" || -z "${JOB_NAME}" ]]; then
  echo "Usage: submit_af3_two_stage.sh <json_path_or_filename> <job_name>"
  exit 1
fi

if [[ "${JSON_INPUT}" = /* ]]; then
  JSON_PATH="${JSON_INPUT}"
elif [[ -f "${JSON_INPUT}" ]]; then
  JSON_PATH="$(realpath "${JSON_INPUT}")"
else
  JSON_PATH="${INPUT_DIR}/${JSON_INPUT}"
fi

if [[ ! -f "${JSON_PATH}" ]]; then
  echo "ERROR: JSON input not found: ${JSON_PATH}"
  exit 1
fi

submit_out="$(
  sbatch "${ARC_SCRIPTS_DIR}/run_af3_data_pipeline.slurm" "${JSON_PATH}"
)"

echo "${submit_out}"
echo
echo "After this file exists:"
echo "  ${OUTPUT_DIR}/${JOB_NAME}/${JOB_NAME}_data.json"
echo
echo "run:"
echo "  sbatch ${ARC_SCRIPTS_DIR}/run_af3_inference.slurm ${JOB_NAME}"
