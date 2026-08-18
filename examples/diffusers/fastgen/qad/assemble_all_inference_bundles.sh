#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -uo pipefail

if (( $# != 1 )); then
    echo "Usage: $0 /path/to/qad_training/output" >&2
    exit 2
fi

OUTPUT_ROOT=$1
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ASSEMBLER=${SCRIPT_DIR}/inference_bundle.py
PYTHON_BIN=${PYTHON_BIN:-python3}

if [[ ! -d "${OUTPUT_ROOT}" ]]; then
    echo "ERROR: output root does not exist: ${OUTPUT_ROOT}" >&2
    exit 2
fi

run_count=0
failure_count=0

for run_dir in "${OUTPUT_ROOT}"/*; do
    [[ -d "${run_dir}/checkpoints" ]] || continue
    ((run_count += 1))
    echo
    echo "[$run_count] Assembling ${run_dir}"
    if ! "${PYTHON_BIN}" "${ASSEMBLER}" "${run_dir}" --materialize symlink; then
        ((failure_count += 1))
        echo "ERROR: assembly failed for ${run_dir}" >&2
    fi
done

echo
echo "Processed ${run_count} run directories; failures=${failure_count}"

if (( failure_count > 0 )); then
    exit 1
fi
