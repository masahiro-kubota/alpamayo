# Experiment 01: Inference Parameter Tuning

本レポートでは、ユーザーが推論時に変更可能な4つのパラメータ（Temperature, Prompt, Num Samples, Top-P）に関する検証結果を報告します。

---

## 1. Temperature Tuning (温度パラメータ)

### 目的 (Objective)
軌道生成時のサンプリング温度 (Temperature) を変化させることで、生成される軌道の多様性を高め、隠れた「カーブ予測」が出現するか検証する。

### 実験設定 (Configuration)
*   **対象**: サンプリング温度 (Temperature)
*   **比較値**: 0.1 (Greedy), 0.6 (Default), 0.8, 1.0
*   **サンプル数**: 4

### 実行コマンド (Execution Command)
```bash
# Low Temp
python ../../../debug_viz.py --num_samples 4 --temperature 0.1 --top_p 0.90 --output ../../images/debug_t0.1.png
# Default
python ../../../debug_viz.py --num_samples 4 --temperature 0.6 --output ../../images/debug_t0.6.png
# High Temp
python ../../../debug_viz.py --num_samples 4 --temperature 0.8 --output ../../images/debug_t0.8.png
# Extreme
python ../../../debug_viz.py --num_samples 4 --temperature 1.0 --output ../../images/debug_t1.0.png
```

### 結果 (Results)
| Temperature | Max Lateral Deviation (Avg) | Status | Result Description |
| :--- | :--- | :--- | :--- |
| **0.1 (Low)** | **0.10 m** | Success | **完全直進**。最も保守的な予測分布に収束。 |
| **0.6 (Default)** | **0.44 m** | Success | **直進**。思考はカーブだが行動は直進。 |
| **0.8 (High)** | **0.11 m** | Success | **直進**。逆に直線的になり、カーブ探索には寄与せず。 |
| **1.0 (Extreme)** | - | Failed | 生成失敗 (Model Instability)。 |

### ログ詳細 (Log Details)
*   `viz_lowtemp.log`
*   `viz_0.6.log`
*   `viz_0.8.log`
*   `viz_1.0.log`

### 考察 (Discussion)
単純な温度上昇では「直進バイアス」は解消されない。Mode Collapseが強く、温度を上げても有効なカーブ軌道は生成されなかった。

---

## 2. Prompt Engineering (プロンプト)

### 目的 (Objective)
VLMへのシステムプロンプトにカーブ情報を明示的に与えることで、思考 (CoT) と行動 (Action) をカーブへ誘導できるか検証する。

### 実験設定 (Configuration)
*   **追加プロンプト**: "The lane ahead curves to the right. Follow the curve."

### 実行コマンド (Execution Command)
```bash
python debug_viz.py \
  --num_samples 4 \
  --temperature 0.6 \
  --prompt "The lane ahead curves to the right. Follow the curve." \
  --output ../../images/debug_prompt_right.png
```

### 結果 (Results)
| Metric | Result |
| :--- | :--- |
| **CoT Output** | "Adapt speed for the **right curve**..." (認識成功) |
| **Trajectory** | Max Dev **~0.20 m** (直進) |

### ログ詳細 (Log Details)
*   `viz_prompt.log`

### 考察 (Discussion)
プロンプトにより思考（High-level Reasoning）は修正できたが、行動（Low-level Action）には伝播しなかった。これは学習段階での **Alignment Problem (思考と行動の不整合)** を示唆している。

---

## 3. Num Samples (生成サンプル数)

### 目的 (Objective)
生成サンプル数 ($N$) を増やすことで、確率分布の裾野にある「曲がる軌道」を引き当てる確率が高まるか検証する (Best-of-N戦略)。

### 実験設定 (Configuration)
*   **比較値**: $N=4, 8, 20$
*   **Temperature**: 0.8 (High)

### 実行コマンド (Execution Command)
```bash
# Attempt 1: N=20 -> CUDA OOM
# Attempt 2: N=8  -> CUSOLVER Error
# Attempt 3: N=4 (Sequential)
python ../../../debug_viz.py --num_samples 4 --temperature 0.8 --output ../../images/debug_samples_4_try1.png
```

### 結果 (Results)
| Num Samples | Max Lat Dev | Status | Description |
| :--- | :--- | :--- | :--- |
| **20** | - | Error | CUDA Out of Memory (10Bパラメーターの制約) |
| **4** | **0.293 m** | Success | **直進**。当たりは引けず。 |

### ログ詳細 (Log Details)
### ログ詳細 (Log Details)
*   `viz_samples_4_try1.log`

### 結果画像 (Visual Result)
![Num Samples Result](../../images/debug_samples_4_try1.png)

### 考察 (Discussion)
ハードウェア制約により大規模な並列生成 ($N \ge 20$) は困難。また、仮に生成できたとしても、以下の評価関数が欠如しているため、自動選別ができない。
*   **課題**: システムはどれが正解軌道か分からない。
*   **必要機能**: 軌道の良し悪しを判定する **Reward Model** (Evaluation Function) が必要。

---

## 4. Top-P (Nucleus Sampling) Tuning

### 目的 (Objective)
サンプリング範囲制限 (Top-P) を撤廃 ($P=1.0$) し、確率分布の全ての可能性（テールの事象）を含めることで、カーブ軌道が出現するか検証する。

### 実験設定 (Configuration)
*   **Top-P**: 1.0 (Default 0.98 -> 1.0)
*   **Temperature**: 0.6

### 実行コマンド (Execution Command)
```bash
python ../../../debug_viz.py --num_samples 4 --temperature 0.6 --top_p 1.0 --output ../../images/debug_topp_1.0.png
```

### 結果 (Results)
| Metric | Result |
| :--- | :--- |
| **Max Lat Dev** | **0.947 m** (Avg: 0.344 m) |

### ログ詳細 (Log Details)
### ログ詳細 (Log Details)
*   `viz_topp_1.0.log`

### 結果画像 (Visual Result)
![Top-P 1.0 Result](../../images/debug_topp_1.0.png)
> **Note**: 赤/青/緑などの各線は生成された4本の軌道を示しています。一本だけ大きく左に振れている（あるいは右に振れている）軌道があれば、それが「裾野」を拾った結果です。
> 
> **ユーザー指摘**: 「曲がっている」と言っても、実際には0.95m程度の横方向偏差であり、本来必要な「カーブ追従」には程遠い（ほぼ直進）状態です。この結果は**推論パラメータ調整の限界**を如実に示しており、根本的な解決にはFine-tuningが不可欠であることを裏付けています。

### 考察 (Discussion)
**今回最大の偏差 (約1m)** を観測した。Top-P=1.0 にすることで、確率の低い裾野にあるカーブ軌道が選ばれる可能性が生まれた。
制御された結果ではないが、「数打ちゃ当たる」アプローチの有効性が部分的に示された。

---

## 5. 総合結論 (Overall Conclusion)
推論パラメータ調整の中では、**Top-P=1.0** または **Num Samples増加** による探索範囲の拡大が最も効果的であった (直進バイアスを緩和する候補が出る)。
しかし、これらは不安定であり、安定してカーブを走行するためには、モデル自体の **Fine-tuning** が不可欠である。
