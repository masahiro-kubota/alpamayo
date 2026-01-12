# Alpamayo R1 を finetuningする (Finetuning Alpamayo R1)

## 1. 目的（問題意識・課題感）
- **課題**: Alpamayo-R1 が単眼カメラ構成などの情報不足時に「頑なに直進する（直進バイアス）」問題を、Diffusion Trajectory Decoder の学習（Fine-tuning）によって解決する。
## TL;DR
**「直進バイアス」の修正に、Cosmos Cookbook は使用できない。**
Cookbook は VLM (Brain) のテキスト生成学習用であり、Alpamayo の直進バイアスの主因である Trajectory Decoder (Diffusion Head) を学習する機能を持たないため。

---

## 1. 目的（問題意識・課題感）
- **What**: カメラ構成実験において、Alpamayo-R1 が単眼カメラ構成などで「頑なに直進する」バイアスを持つことが判明した。これを解決するためにファインチューニングを行いたい。
- **Why**: 
    - Alpamayo-R1 は Cosmos Reason (Reason 1) をバックボーンとして使用している。
    - 公式の **Cosmos Cookbook** には Post-Training (SFT) のレシピがあり、これが使えるのではないかと考えた。
- **Goal**: Cosmos Cookbook の仕様と Alpamayo-R1 の構造を照らし合わせ、**「今回の課題（軌道修正）に Cookbook が使えるか？」** を明確に結論付ける。

## 2. Approach

以下の3段階で構造的な適合性を検証しました。

1.  **Framework Analysis**: Cosmos Cookbook が対象としているモデル (Reason 1/2) の出力形式を確認する。
2.  **Architecture Alignment**: Alpamayo-R1 の内部構造（特に軌道生成部）と、Cookbook の学習対象を突合する。
3.  **Feasibility Check**: 今回の問題である「Trajectory バイアス」が、Cookbook の守備範囲内か判定する。

## 3. 前提環境 (Prerequisites)
- **Repo**: [nvidia-cosmos/cosmos-cookbook](https://github.com/nvidia-cosmos/cosmos-cookbook)
- **Target Model Structure**: Alpamayo-R1 (Cosmos-Reason Backbone + Diffusion Head)

## 4. 具体的な検証手順 (Concrete Steps)

### Step 1: Cosmos Cookbook の対象モデル (Target Models)

Cosmos Cookbook は多様なモデルを扱っており、厳密には以下の3種類に大別されます。

1.  **Cosmos Reason 2**:
    *   **Architecture**: `Qwen3-VL-8B` ベースの VLM。
    *   Cookbook内の `reason2/` ディレクトリがこれに該当します。
2.  **Cosmos Reason 1**:
    *   **Architecture**: `Qwen2.5-VL` ベースの VLM。
    *   **重要:** **Alpamayo-R1 のバックボーンはこちらです。**
    *   Cookbook内の `reason1/` ディレクトリがこれに該当します。
3.  **Cosmos Predict / Transfer**:
    *   **Architecture**: VLMではなく、画像生成・動画生成モデル (Generative World Model)。Qwenベースではありません。

**結論:**
Cookbook は Qwen3ベース (Reason 2) も Qwen2.5ベース (Reason 1) も対象にしています。
Alpamayo-R1 のバックボーンを改善したい場合は、**`reason1/` (Qwen2.5-VL向け)** のレシピが直接的な参考になります。

ただし、これら **Reason モデル (VLM)** に共通する重要な仕様として、以下が挙げられます（これがAlpamayoのDiffusion Headと整合しない理由です）。

*   **Output Modality**:
    *   すべて **「テキスト (Tokens)」** です。
    *   Alpamayoのような「連続値の軌道」ではなく、推論結果として `{"trajectory": [[1.5, 2.0], ...]}` のような **JSON文字列** を出力します。
*   **Training Objective**:
    *   したがって、Cookbook の Post-Training (SFT) は、**「Next Token Prediction (Cross Entropy Loss)」だけ** を最小化するように設計されています。

### Step 2: Alpamayo-R1 の構造分解 (Structural Analysis)

次に、Alpamayo-R1 のアーキテクチャを詳細に分解します。
Alpamayo は単なる VLM ではなく、**「脳 (VLM) + 手足 (Diffusion)」の複合アーキテクチャ** です。

![Alpamayo Architecture](/home/masa/.gemini/antigravity/brain/8959761e-8e7c-43be-86fc-804e07b18bfc/uploaded_image_1768134338049.png)

#### 1. 脳: Cosmos-Reason Backbone (Distilled)
*   **実体:** `Cosmos Reason 1` (Qwen2.5-VLベースのPhysical AIモデル)
*   **役割:** 画像認識、状況理解、思考 (CoT) の生成。
*   **学習の経緯:**
    *   巨大な教師モデル (`Qwen2-VL-72B`) が生成した「高品質な思考テキスト(Reasoning Traces)」を蒸留 (Distillation) して作られました。
    *   これにより、8Bという軽量サイズながら、72Bクラスの論理的思考力を持っています。
*   **Cookbookとの関係:**
    *   この部分は純粋な VLM であるため、**Cosmos Cookbook の対象範囲と完全に一致します。**

#### 2. 手足: Diffusion Trajectory Decoder
*   **実体:** `Diffusion Model` (Flow Matching)
*   **役割:** 実際のハンドル操作・走行軌道 (Trajectory) の生成。
*   **仕組み (ここが重要):**
    *   VLMがテキストを出力するのではなく、**VLMの最終隠れ層 (Hidden States)** を入力として受け取ります。
    *   そこから **拡散過程 (Denoising)** を経て、連続値としての滑らかな軌道を生成します。
    *   テキスト生成 (Next Token Prediction) とは全く異なる数学的プロセス (MSE / Flow Matching Loss) で動いています。
*   **Cookbookとの関係:**
    *   **Cookbook の学習ループには、この拡散モデルを学習するための機能が一切含まれていません。**
    *   したがって、Cookbook で学習しても、この「手足」は一切更新されず、両者の連携が取れなくなる恐れがあります。

### Step 3: 学習可能性の判定 (Feasibility Mapping)

「直進バイアスを直したい」という目的に対して、Cookbook が機能するかを判定します。

*   **CoT (思考) の修正**: **可能**
    *   「カーブがあることを認識させる」ために VLM 部分を再学習することは、Cookbook で可能です。
    *   学習データ生成プロセスにおいて、Teacher (Qwen-72B) から蒸留された「正しい推論」を VLM に教えることができます。
*   **Trajectory (軌道) の修正**: **不可能**
    *   直進バイアスこの Head を再学習させるには、Cookbook ではなく、**Trajectory Loss を扱える Alpamayo-R1 独自の学習コード（DeepSpeed実装）** を使用する必要があります。

#### Q. Trajectory Decoder の追加学習で、単眼でも適切な軌道を出力できるか？
**Answer: はい、理論上は可能です（それが本命の解決策です）。**

*   **理由**: 現在の「直進バイアス」は、モデルが「360度カメラがある前提」の学習済み重みを持っており、単眼（前方のみ）の入力だと情報不足で「安全側に倒して直進する」ような分布を学習してしまっている可能性があります。
*   **解決策**: 「単眼画像 (Front Camera)」と「正解の曲がる軌道 (Ground Truth Trajectory)」のペアだけを集めたデータセットで Trajectory Decoder（拡散モデル）を再学習（Fine-tuning）させます。
*   **効果**: これにより、拡散モデルは「前方画像の特徴（カーブの白線など）」**だけ** を手がかりにして、適切な曲率の軌道を生成する分布 `p(trajectory | single_image)` を獲得できるようになります。

---

## (Appendix) おまけ: Cosmos Cookbook でファインチューニングを実行するには、**Diffusion Loss (MSE / Flow Matching Loss) を計算する機能も、Diffusion Head の重みを更新する機能もありません。**

## 5. 結果のまとめ (Results Summary)

検証の結果、Cosmos Cookbook の適用可能性は以下の通りとなりました。

| 対象モジュール | 役割 | Cosmos Cookbook で学習可能？ | 理由 |
| :--- | :--- | :--- | :--- |
| **Cosmos-Reason Backbone** | 画像認識・思考・CoT記述 | **Yes** (SFT Recipe) | VLMのテキスト生成タスクであるため。 |
| **Trajectory Decoder** | **実走行軌道の生成 (今回の課題)** | **No** | **拡散モデルであり、Cookbookの学習範疇外のため。** |

### (Solution) Trajectory Decoder の追加学習手順
Cookbook は使用せず、今回作成した独自の学習スクリプトを用いて学習を実施します。

- **解決策**: 「前方画像」と「正解の旋回軌道」をペアにして拡散モデルを再学習させ、視覚情報（白線等）から適切な曲率を導き出せるように矯正する。

---

## Part 2: 実装・実行ガイド (Implementation Guide)

### 1. 実験ディレクトリ構成 (Directory Structure)

全ての成果物は `trajectory_bias_experiment/` 配下に集約して管理します。

```text
trajectory_bias_experiment/
├── data/           # 実験専用に抽出されたデータセット (*.pt)
├── checkpoints/    # 学習済みチェックポイント (*.pt)
├── logs/           # 推論結果のメタデータ付き詳細ログ (*.json)
├── images/         # 比較レポート・可視化画像 (*.png)
└── documents/      # 実験ノート・ドキュメント (finetuning.md 等)
```

### 2. 環境構築と要件 (Step 0: Environment Setup)

> [!CAUTION]
> **24GB GPUでは学習不可能 (2025-01-12 検証済み)**
>
> 当初は「VLM凍結・Headのみ学習」のため24GB (RTX 4090等) で足りると見込んでいましたが、**実際には不足します**。
>
> **理由**: Trajectory Decoderは、学習時に **VLMのForward Pass** を実行して `past_key_values` (KVキャッシュ) を取得し、それをExpert Modelに渡す必要があります。
> つまり「VLMのForward + Expert ModelのForward + Backward」を1ステップで行うため、VLMの重み (20GB) + 活性化メモリ (KVキャッシュ、中間テンソル) が全て必要です。
>
> - **実測値 (batch_size=1)**: 約23.5GB → **24GB GPUでOOM**
> - **推奨**: **48GB以上** (A6000, L40S等) または **80GB** (A100, H100)
> - **代替案**:
>   - **VLM出力の事前計算**: VLMのforward結果をディスクにキャッシュし、学習時はExpert+Diffusion Headのみ実行
>   - **Gradient Checkpointing**: メモリ削減 (30-50%減)、ただし学習速度低下
>   - **DeepSpeed ZeRO-3 / FSDP**: マルチGPUでメモリ分散

**必要なGPUメモリ (VRAM): 48GB以上推奨** (A6000, L40S, A100等)
**必要なストレージ容量: 100GB以上** (データセットキャッシュ + モデル重み + 保存用)

> **メモリ計算の根拠 (更新版)**:
> *   **モデル重み (VLM + Expert + Heads)**: 約20GB (10Bパラメータ × 2バイト[bfloat16])
> *   **KVキャッシュ + 中間活性化**: 約15-20GB (VLM Forward時の中間テンソル)
> *   **Backward用の勾配**: 約5-10GB (Headパラメータのみだが、計算グラフが大きい)
> *   **合計**: 約40-50GB → **24GBでは不足**
>
> **ストレージ計算の根拠**:
> *   **データセット**: 約50GB
>     *   1000クリップ (約20秒/個, 高画質) の実容量 ≈ 25GB
>     *   Streaming時のShard単位キャッシュ等のオーバーヘッド ≈ 25GB
> *   **その他**: モデル重み(20GB) + チェックポイント保存領域(数GB)
> *   ※ 全データ(80k時間)を一括ダウンロードする場合は数TB必要ですが、Streamingなら上記で足ります。

GPU環境にて、以下の手技でセットアップを行います。
`uv` をお使いの場合は、`pyproject.toml` の同期後に不足しているライブラリを追加します。

```bash
# uv を使用する場合 (推奨)
# 基本ライブラリは pyproject.toml に含まれていますが、datasets のみ追加が必要です。
uv add datasets

# パスの設定 (srcをPYTHONPATHに追加)
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
```

### 3. 今回の検証手順: パイロットラン (Pilot Run)

本格的な学習の前に、少量のデータで「直進バイアスの解消」が視覚的に確認できるか検証します。

### Step 1: 高曲率クリップの特定と抽出
データセットから「急カーブ」を自動検出し、評価用ベンチマークセットを作成します。

- **出力物:**
    - `trajectory_bias_experiment/data/eval_large_curve_full.pt`: 急カーブ（理想・4カメ全て表示）
    - `trajectory_bias_experiment/data/eval_large_curve_masked.pt`: 急カーブ（現状・左右カメラを黒塗り）

```bash
# 1-1. 急カーブの特定 (最初の500クリップをスキャン)
python scan_all_curves.py --threshold 0.05 --max_clips 500 \
    --output trajectory_bias_experiment/logs/curve_scan_500samples.json

# 1-2. 上位5件のIDを使用してデータを抽出 (本当のカーブ地点を自動でサンプリング)
CLIP_IDS="09312f4a-c618-46a8-a8ff-1db37e043b5d,297615c4-dae9-40b6-9051-309ef3dcb02d,559ac095-5325-4a2d-a13c-08944d35d106,df0687d0-a2a3-4534-95e0-164f02bec8af,7a22a3d6-1156-427f-b8d6-ac379c6f9acb"

python scripts/finetuning/prepare_training_data.py \
    trajectory_bias_experiment/data/eval_large_curve_full.pt --no_mask --clip_ids "$CLIP_IDS"

python scripts/finetuning/prepare_training_data.py \
    trajectory_bias_experiment/data/eval_large_curve_masked.pt --clip_ids "$CLIP_IDS"
```

### Step 2: 訓練用および通常評価用のデータ準備
上記ベンチマーク以外のクリップから、動作確認用のセットを作成します。

- **出力物:**
    - `trajectory_bias_experiment/data/train_debug.pt`: 訓練データ (10サンプル)
    - `trajectory_bias_experiment/data/val_debug.pt`: 評価データ (1サンプル)

```bash
python scripts/finetuning/prepare_training_data.py trajectory_bias_experiment/data/train_debug.pt --samples 10 --split train
python scripts/finetuning/prepare_training_data.py trajectory_bias_experiment/data/val_debug.pt --samples 1 --split val
```

> [!NOTE]
> **訓練データの作成方法 (Data Curation Strategy)**
>
> 現在の実装 (`prepare_training_data.py`) と論文の手法には差異があります。
>
> **1. 現在の実装 (Pilot Implementation)**
> 20秒のクリップ内で **「曲率 (Curvature) が最大になる瞬間」** を1点だけ特定し、その前後を切り出して学習データとします。
> これは「直進バイアス」を解消するために、最もカーブがきつい場面を優先的に学習させるための簡易的な戦略です。
>
> **2. Alpamayo-R1 論文の手法 (Ideal Strategy)**
> 論文では、単にカーブだけでなく「意思決定の瞬間」をすべて抽出します。
>
> *   **Reactive Scenarios (即応的判断)**:
>     > "a keyframe is typically chosen by applying a short temporal buffer (approximately **0.5 seconds**) before the ego vehicle initiates a behavior change corresponding to a driving decision."
>     > (自車が運転判断に対応する挙動変化を開始する約0.5秒前をキーフレームとして選択する)
>
> *   **Auto-Labeling (自動抽出)**:
>     > "treat the frame at which a **meta action transition occurs** as a decision-making moment, allowing us to determine the keyframe automatically and efficiently across large scale data."
>     > (メタアクションの遷移が発生するフレームを意思決定の瞬間として扱い、大規模データから効率的にキーフレームを決定する)

### Step 2.5: 学習前のベースライン評価 (Baseline Evaluation)
比較の基準となる「未学習モデル」の結果を JSON (メタデータ付き) で取得します。

- **出力物 (JSONログ):**
    - `trajectory_bias_experiment/logs/eval_val_debug_baseline.json`: 通常Valのベースライン
    - `trajectory_bias_experiment/logs/eval_baseline_full.json`: 急カーブ(理想)の結果
    - `trajectory_bias_experiment/logs/eval_baseline_masked.json`: 急カーブ(現状)の結果

```bash
# 通常評価用
python scripts/finetuning/evaluate_trajectory_decoder.py \
    --data_path trajectory_bias_experiment/data/val_debug.pt \
    --ckpt_path None \
    --output_json trajectory_bias_experiment/logs/eval_val_debug_baseline.json

# 高曲率ベンチマーク用 (統合スクリプト)
python scripts/finetuning/run_comparison_benchmark.py \
    --full_data trajectory_bias_experiment/data/eval_large_curve_full.pt \
    --masked_data trajectory_bias_experiment/data/eval_large_curve_masked.pt \
    --ckpt_path None \
    --output_dir trajectory_bias_experiment
```

### Step 3: パイロット学習の実行 (Training)
Diffusion Head のみを再学習させます。

- **出力物:**
    - `trajectory_bias_experiment/checkpoints/ckpt_epoch_9.pt`

```bash
python scripts/finetuning/train_trajectory_decoder.py \
    --data_path trajectory_bias_experiment/data/train_debug.pt \
    --output_dir trajectory_bias_experiment/checkpoints \
    --epochs 10 \
    --batch_size 1
```

### Step 4: 学習後の評価実行 (FT Evaluation)
統合スクリプトは、Step 2.5 で作成されたベースライン JSON を自動で検出し、FT の結果だけを追加推論して比較します。

- **出力物 (JSONログ):**
    - `trajectory_bias_experiment/logs/eval_val_debug_ft.json`: 通常Valの改善後
    - `trajectory_bias_experiment/logs/eval_ft_ckpt_epoch_9.json`: 急カーブの改善後

```bash
# 通常評価
python scripts/finetuning/evaluate_trajectory_decoder.py \
    --data_path trajectory_bias_experiment/data/val_debug.pt \
    --ckpt_path trajectory_bias_experiment/checkpoints/ckpt_epoch_9.pt \
    --output_json trajectory_bias_experiment/logs/eval_val_debug_ft.json

# 高曲率ベンチマーク
python scripts/finetuning/run_comparison_benchmark.py \
    --full_data trajectory_bias_experiment/data/eval_large_curve_full.pt \
    --masked_data trajectory_bias_experiment/data/eval_large_curve_masked.pt \
    --ckpt_path trajectory_bias_experiment/checkpoints/ckpt_epoch_9.pt \
    --output_dir trajectory_bias_experiment
```

### Step 5: 推論結果の可視化 (Visualization)
JSON に保存されたメタデータと座標データを元に、比較レポート画像を生成します。

- **出力物 (PNG画像):**
    - `trajectory_bias_experiment/images/val_debug_comparison.png`: 通常シーン(1枚)の前後比較
    - `trajectory_bias_experiment/images/benchmark_vs_ckpt_epoch_9.png`: 急カーブ(5枚)の3構成比較

```bash
# 5-1. 通常シーンの比較
python scripts/finetuning/visualize_eval_results.py \
    --results_jsons trajectory_bias_experiment/logs/eval_val_debug_baseline.json,trajectory_bias_experiment/logs/eval_val_debug_ft.json \
    --labels "Original","Fine-tuned" \
    --data_path trajectory_bias_experiment/data/val_debug.pt \
    --output_file trajectory_bias_experiment/images/val_debug_comparison.png

# 5-2. 急カーブベンチマークの比較
python scripts/finetuning/visualize_eval_results.py \
    --results_jsons trajectory_bias_experiment/logs/eval_baseline_full.json,trajectory_bias_experiment/logs/eval_baseline_masked.json,trajectory_bias_experiment/logs/eval_ft_ckpt_epoch_9.json \
    --labels "Original (Full)","Original (Masked)","FT-Pilot" \
    --data_path trajectory_bias_experiment/data/eval_large_curve_full.pt \
    --output_file trajectory_bias_experiment/images/benchmark_vs_ckpt_epoch_9.png
```

---

### 4. 基本的な実行手順 (Standard Workflow)

Hugging Face 上の `nvidia/PhysicalAI-Autonomous-Vehicles` データセットから、学習データを抽出・作成します。

#### Step 1: データセットの準備 (Data Prep)

> [!TIP]
> **パイプラインの動作確認 (Pilot Run)**
> いきなり全データを処理するのではなく、まずは少量のデータでエラーなくパイプラインが通るか確認することを強く推奨します。
> 詳細は前述の「パイロットラン」セクションを参照してください。

本番データの作成例:
```bash
# 学習データ (Train Split: 90%)
# --samples 1000 は「1000クリップ (約5.5時間分)」を意味します
python scripts/finetuning/prepare_training_data.py data/training_data.pt --samples 1000 --split train

# 評価データ (Val Split: 10%)
python scripts/finetuning/prepare_training_data.py data/val_data.pt --samples 100 --split val
```

*   **処理内容**: 指定した Split に属するデータを抽出し、.pt ファイルを作成します。
*   **Splitting**: Clip IDのハッシュ値(0-99)に基づき、0-89をTrain、90-99をValとして分割します。

#### Step 2: 学習の実行 (Training)

Trajectory Decoder (Diffusion Head) のみを再学習させます。

```bash
# 使用法: python scripts/finetuning/train_trajectory_decoder.py --data_path <pt_file>
python scripts/finetuning/train_trajectory_decoder.py \
    --data_path data/training_data.pt \
    --output_dir ./checkpoints \
    --epochs 10 \
    --batch_size 4
```

*   **VLM Backbone (Freeze)**: 画像・言語理解能力は維持。
*   **Diffusion Head (Train)**: Flow Matching Loss (MSE) を最小化して学習。

#### Step 3: 評価の実行 (Evaluation)

学習した重みをロードし、未学習のテストデータ（カーブ等）に対して推論を行います。

```bash
# 使用法: python scripts/finetuning/evaluate_trajectory_decoder.py --data_path <val_pt_file> --ckpt_path <ckpt_file>
python scripts/finetuning/evaluate_trajectory_decoder.py \
    --data_path data/val_data.pt \
    --ckpt_path ./checkpoints/ckpt_epoch_9.pt \
    --output_dir ./eval_results \
    --num_samples 10
```

1.  **評価の仕組み**:
    *   ベースモデル (`nvidia/Alpamayo-R1-10B`) をロード。
    *   指定されたチェックポイントから **Diffusion Headの重みのみ** を上書きロード。
    *   推論結果を描画して保存。

---

## 5. 本番学習の実施 (Full Fine-tuning)

パイロット学習で「曲がれる」ようになったことを確認後、規模を拡大します。

1.  **データ準備 (1000サンプル)**: `trajectory_bias_experiment/data/training_data_full.pt`
2.  **本番学習 (10エポック)**: チェックポイントを `trajectory_bias_experiment/checkpoints/full_ft/` に保存。
3.  **ベンチマーク評価**: `run_comparison_benchmark.py` を使用して、全クリップでの ADE/FDE などの傾向を視覚的に確認。

---

## 6. 補足: JSON メタデータの活用
生成される JSON には、以下の情報が自動的に記録されます。
- `model_id`: ベースとしたモデル名
- `ckpt_path`: 使用した学習済み重み
- `data_path`: 推論に使用したデータセット
- `parameters`: 推論時の Temperature 等のパラメータ

`run_comparison_benchmark.py` はこの情報をチェックするため、**同じデータ・同じ重みでの評価が既にある場合は推論をスキップ** し、一瞬で可視化フェーズに移行できます。

---

## 7. トラブルシューティング (Troubleshooting)

### Q: `RuntimeError: PytorchStreamWriter failed` during save
チェックポイント保存時に発生する場合、GPUメモリ上のテンソルを直接保存しようとして同期エラーやディスクI/Oのタイムアウトが起きている可能性があります。
`torch.save` の前に `v.cpu()` で重みをCPUに移動させてください。

### Q: `TypeError: linspace() ...` in Qwen3-VL forward
バッチサイズ > 1 の学習時に発生します。`pixel_values` や `image_grid_thw` を `torch.stack` ではなく `torch.cat` で結合する必要があります（Qwen3-VLの実装依存）。

### Q: `AttributeError: 'Interpolator' object has no attribute ...`
`get_clip_feature` で取得する `egomotion` オブジェクトの属性は以下のようにアクセスします：
- 曲率生データ: `egos.curvature` (NumPy Array)
- 速度など: `egos.velocity` (NumPy Array)
**`.linear` などの属性は存在しません。**

### Q: `RuntimeError: mat1 and mat2 must have the same dtype`
モデルが `bfloat16` でロードされている場合、Diffusionに入力する `x_t` や `timesteps` も明示的に `.to(dtype=model.dtype)` でキャストする必要があります。

### Q: `RuntimeError: The size of tensor a (2) must match ... b (3)`
モデルの出力次元（2D: x,y）とデータの次元（3D: x,y,z）が不一致の場合に発生します。loss計算前にターゲット `u_t` を `u_t[..., :out_dim]` のようにスライスして次元を合わせてください。

### Q: `TypeError: ... missing 1 required positional argument: 'timesteps'`
`action_in_proj` (PerWaypointActionInProjV2) には `timesteps` 引数が必要です。`(B, 1)` の形状で `timesteps=t` として渡してください。
