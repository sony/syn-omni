#!/usr/bin/env bash
# Shared base-directory configuration for scripts/*.sh.
# Override via environment variables, e.g.:
#   DATA_ROOT=/my/data OUTPUT_BASEDIR=/my/output ./scripts/syn_omni_3b.sh

DATA_ROOT="${DATA_ROOT:-/home/appuser/datasets}"
OUTPUT_BASEDIR="${OUTPUT_BASEDIR:-/home/appuser/omni/outputs}"
