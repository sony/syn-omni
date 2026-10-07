#!/usr/bin/env bash
# Shared base-directory configuration for scripts/*.sh.
# Override via environment variables, e.g.:
#   DATA_BASEDIR=/my/data OUTPUT_BASEDIR=/my/output ./scripts/baselines/e5_omni_3b.sh

DATA_BASEDIR="${DATA_BASEDIR:-/home/appuser/datasets/vlm2vec_eval}"
OUTPUT_BASEDIR="${OUTPUT_BASEDIR:-/home/appuser/datasets/embedding_results}"
