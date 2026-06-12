#!/bin/bash
# Default PANDORA database location on Longleaf (/work, not home quota).
# Override: export PANDORA_DB_ROOT=/your/path

: "${PROJECT_ROOT:=${PMGEN_ROOT%/PMGen}}"
export PANDORA_DB_ROOT="${PANDORA_DB_ROOT:-${PROJECT_ROOT}/PANDORA_databases/default}"
