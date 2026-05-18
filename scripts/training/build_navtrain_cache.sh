#!/bin/bash
# Build navtrain feature cache for flow_rl_agent training
# Uses Ray workers (all available CPUs) to cache TransfuserFeatures for all navtrain tokens
# Expected time: ~1-3 hours on 16-core CPU

set -e

export PYTHONPATH=$NAVSIM_DEVKIT_ROOT:$PYTHONPATH

echo "[$(date)] Starting navtrain feature cache building..."
echo "Output dir: $NAVSIM_EXP_ROOT/training_cache"
echo "Logs: $OPENSCENE_DATA_ROOT/navsim_logs/trainval"
echo "Sensor blobs: $OPENSCENE_DATA_ROOT/sensor_blobs/trainval"

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_dataset_caching.py \
    agent=transfuser_agent \
    experiment_name=navtrain_cache_build \
    train_test_split=navtrain \
    cache_path=$NAVSIM_EXP_ROOT/training_cache \
    force_cache_computation=false \
    'worker.threads_per_node=4'

echo "[$(date)] navtrain feature cache building COMPLETE."
echo "Cached to: $NAVSIM_EXP_ROOT/training_cache"
