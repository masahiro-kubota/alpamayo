#!/bin/bash
# Enable venv
source ar1_venv/bin/activate

# Optimize memory
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Starting Sweep ==="

echo "[1/3] Running temp=0.6..."
python debug_viz.py --num_samples 4 --temperature 0.6 --output debug_t0.6.png > viz_0.6.log 2>&1
if [ $? -eq 0 ]; then echo "✓ 0.6 Success"; else echo "✗ 0.6 Failed"; fi

echo "[2/3] Running temp=0.8..."
python debug_viz.py --num_samples 4 --temperature 0.8 --output debug_t0.8.png > viz_0.8.log 2>&1
if [ $? -eq 0 ]; then echo "✓ 0.8 Success"; else echo "✗ 0.8 Failed"; fi

echo "[3/3] Running temp=1.0..."
python debug_viz.py --num_samples 4 --temperature 1.0 --output debug_t1.0.png > viz_1.0.log 2>&1
if [ $? -eq 0 ]; then echo "✓ 1.0 Success"; else echo "✗ 1.0 Failed"; fi

echo "=== Sweep Complete ==="
