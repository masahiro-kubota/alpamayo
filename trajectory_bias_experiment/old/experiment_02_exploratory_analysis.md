# 実験レポート 02: 探索的分析 (Exploratory Analysis)

## 目的 (Objective)
パラメータ調整以外の方法（プロンプト指示、極端なパラメータ設定、入力形式変更）によって、モデルの直進バイアスを打破できるか、またはバイアスの性質を特定する。

## 実験設定 (Configuration)
*   **対象**: Prompt Engineering, Low Temperature, Multi-Camera Simulation
*   **共通設定**: Rosbag `rosbag2_autoware_0.mcap` (Ratio 0.6)

## 1. プロンプトエンジニアリング (Prompt Engineering)
*   **仮説**: CoT（思考）に介入することで、軌道生成を誘導できるか。
*   **コマンド**:
    ```bash
    python debug_viz.py \
      --num_samples 4 \
      --temperature 0.6 \
      --prompt "The lane ahead curves to the right. Follow the curve." \
      --output debug_prompt_right.png
    ```
*   **結果**:
    *   **CoT**: "Adapt speed for the right curve..." （認識成功）
    *   **軌道**: 直進 (Max Dev ~0.20m)
    *   **ログ参照**: `viz_prompt.log`

## 2. 低温度実験 (Low Temperature)
*   **仮説**: モデルが「最も確率が高い」と考えているデフォルトの軌道を確認する。
*   **コマンド**:
    ```bash
    python debug_viz.py \
      --num_samples 4 \
      --temperature 0.1 \
      --top_p 0.90 \
      --output debug_t0.1.png
    ```
*   **結果**:
    *   **軌道**: 完全な直進 (Max Dev ~0.10m)
    *   **ログ参照**: `viz_lowtemp.log`

## 3. マルチカメラシミュレーション (Multi-Camera)
*   **仮説**: 学習時と同じ3カメラ入力（左右黒画像）にすることで、Tensor形状起因の問題を解消できるか。
*   **コマンド**:
    ```bash
    python debug_viz.py \
      --multicam \
      --num_samples 1 \
      --output debug_multicam.png
    ```
*   **結果**:
    *   **軌道**: 直進 (Max Dev ~0.27m)
    *   **ログ参照**: `viz_multicam_retry.log`

## 考察 (Discussion)
いずれのアプローチでも「直進バイアス」は強固であった。特にプロンプト実験において、思考（CoT）レベルではカーブを認識できているにもかかわらず、行動（軌道）出力に反映されないことから、Diffusion Head部分が視覚情報（または地図の欠落）に強く依存して保守的な出力をしていることが特定された。
