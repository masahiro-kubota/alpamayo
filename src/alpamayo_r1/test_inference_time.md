# Alpamayo R1 Inference Timing Results

## Experiment Setup
- **Date**: 2026-01-16
- **Model**: `nvidia/Alpamayo-R1-10B`
- **Device**: NVIDIA GeForce RTX 4090
- **Precision**: `bfloat16`
- **Strategy**: 3 Warmup iterations, 1 Measurement run

## Timing Breakdown (Hot Run)

| Phase | Time (ms) |
| :--- | :--- |
| **Prefilling (+Vision)** | **506.04 ms** |
| **Reasoning Decoding** | **607.39 ms** |
| **Trajectory Decoding (Flow)** | **354.58 ms** |
| Overhead | 0.72 ms |
| **Total End-to-End** | **1468.74 ms** |

## Notes
- **Warmup effect**: First run took ~3.5s, subsequent runs ~1.4s.
- **Comparison to Paper (99ms)**:
    - Current result (~1.4s) is still ~14x slower.
    - **Vision/Prefill** (506ms vs 20ms): Major bottleneck. The local setup might be processing high-res video or using unoptimized attention kernels compared to the paper's setup (likely H100 with specialized kernels).
    - **Trajectory Decoding** (354ms vs 8.75ms): Significant difference. The paper mentions "5 steps" taking 8.75ms, implying ~1.75ms/step. We are taking ~70ms/step. This suggests we might be running more steps or lacking `torch.compile` / TensorRT optimizations.
