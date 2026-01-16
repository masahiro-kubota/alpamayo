# Alpamayo R1 Inference Timing Results

## Experiment Setup
- **Date**: 2026-01-16
- **Model**: `nvidia/Alpamayo-R1-10B`
- **Device**: NVIDIA GeForce RTX 4090
- **Precision**: `bfloat16`

## Timing Breakdown

| Phase | Time (ms) |
| :--- | :--- |
| **Prefilling (+Vision)** | **2946.19 ms** |
| **Reasoning Decoding** | **869.49 ms** |
| **Trajectory Decoding (Flow)** | **463.94 ms** |
| Overhead | 19.59 ms |
| **Total End-to-End** | **4299.21 ms** |

## Notes
- Measured using `src/alpamayo_r1/test_inference.py`.
- **Prefilling** includes the Vision Encoder execution and the initial prompt processing.
- **Reasoning Decoding** covers the generation of chain-of-thought and other text tokens.
- **Trajectory Decoding** is the diffusion process for generating the trajectory.
