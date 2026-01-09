# 実験レポート 01: パラメータスイープ (Temperature Sweep)

## 目的 (Objective)
軌道生成時のサンプリング温度（Temperature）を変化させることで、生成される軌道の多様性を高め、隠れた「カーブ予測」が出現するかどうかを検証する。

## 実験設定 (Configuration)
*   **対象バグ**: Rosbag `rosbag2_autoware_0.mcap` (Ratio 0.6)
*   **サンプル数**: 4
*   **比較パラメータ**: Temperature = 0.6 (Default), 0.8, 1.0

## 実行コマンド (Execution Command)
```bash
# 自動スイープスクリプトの実行
bash run_debug_sweep.sh

# または手動実行
# python debug_viz.py --num_samples 4 --temperature 0.6 --output debug_t0.6.png
# python debug_viz.py --num_samples 4 --temperature 0.8 --output debug_t0.8.png
```

## 結果 (Results)

| Temperature | Max Lateral Deviation (Avg) | Status | Result Description |
| :--- | :--- | :--- | :--- |
| **0.6** | ~0.44 m | Success | **直進**。偏差は誤差範囲で、明確なカーブは生成されず。 |
| **0.8** | ~0.11 m | Success | **直進**。むしろ偏差が減少し、より直線的になった。 |
| **1.0** | - | Failed | 生成失敗（モデルの不安定化）。 |

### ログ詳細 (Log Details)
*   `viz_0.6.log`
*   `viz_0.8.log`
*   `viz_1.0.log`

<details><summary>Output Example (Temp 0.6)</summary>

```text
  Trajectory Statistics (Local Frame):
  ID | Max Lat Dev (m) | Final Y (m) | Curvature Score
  ---|-----------------|-------------|----------------
   0 |         0.143   |      -0.143 | N/A
   1 |         0.233   |      -0.233 | N/A
   2 |         0.999   |      -0.999 | N/A
   3 |         0.389   |       0.389 | N/A

  [Summary] Avg Max Lat Dev: 0.441 m, Max of all: 0.999 m
```
</details>

## 考察 (Discussion)
単純なパラメータ温度の上昇では、モデルの「直進バイアス」を解消することはできなかった。むしろ温度を上げてもカーブが出現しないことから、モデルの分布自体が直進に極端に偏っていることが示唆される。