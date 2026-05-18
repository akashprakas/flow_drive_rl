#!/bin/bash
# Run 2-epoch FlowRL navtrain training with per-epoch checkpoints
# Prerequisites: navtrain training_cache must be built first (run build_navtrain_cache.sh)

set -e

export PYTHONPATH=$NAVSIM_DEVKIT_ROOT:$PYTHONPATH

# Verify cache exists before starting
if [ ! -d "$NAVSIM_EXP_ROOT/training_cache" ]; then
    echo "ERROR: training_cache not found at $NAVSIM_EXP_ROOT/training_cache"
    echo "Run build_navtrain_cache.sh first to build the feature cache."
    exit 1
fi

CACHE_COUNT=$(find $NAVSIM_EXP_ROOT/training_cache -name "transfuser_feature.gz" | wc -l)
echo "[$(date)] Found $CACHE_COUNT cached samples in training_cache"
echo "[$(date)] Starting FlowRL navtrain training (2 epochs, per-epoch checkpoints)..."

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
    --config-name flow_rl_navtrain

echo "[$(date)] Training complete."
