#!/bin/bash


export OPENSCENE_DATA_ROOT=/home/akash/learn/navsim/download
export NAVSIM_EXP_ROOT=/home/akash/learn/navsim/exp
export NAVSIM_DEVKIT_ROOT=/home/akash/learn/navsim
export NUPLAN_MAPS_ROOT=${OPENSCENE_DATA_ROOT}/maps


mkdir -p "$NAVSIM_EXP_ROOT"

echo "Environment variables set:"
echo "  OPENSCENE_DATA_ROOT = $OPENSCENE_DATA_ROOT"
echo "  NAVSIM_EXP_ROOT     = $NAVSIM_EXP_ROOT"
echo "  NAVSIM_DEVKIT_ROOT  = $NAVSIM_DEVKIT_ROOT"
echo "  NUPLAN_MAPS_ROOT    = $NUPLAN_MAPS_ROOT"
