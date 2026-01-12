# Alpamayo R1 を finetuningする (Finetuning Alpamayo R1)

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

#### Step 0: 環境構築 (Environment Setup)

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

#### Step 1: データセットの準備 (Data Prep)
Hugging Face 上の `nvidia/PhysicalAI-Autonomous-Vehicles` データセットから、学習データを抽出・作成します。

> [!TIP]
> **パイプラインの動作確認 (Pilot Run)**
> いきなり全データを処理するのではなく、まずは少量のデータでエラーなくパイプラインが通るか確認することを強く推奨します。
> ```bash
> # 動作確認用: 10サンプルのみ抽出
> python scripts/finetuning/prepare_training_data.py data/debug.pt --samples 10 --split train
> 
> # 動作確認用: 1エポックだけ学習
> python scripts/finetuning/train_trajectory_decoder.py --data_path data/debug.pt --output_dir ./debug_ckpt --epochs 1 --batch_size 2
> ```

本番データの作成:
```bash
# 学習データ (Train Split: 90%)
# --samples 1000 は「1000クリップ (約5.5時間分)」を意味します
python scripts/finetuning/prepare_training_data.py data/training_data.pt --samples 1000 --split train

# 評価データ (Val Split: 10%)
python scripts/finetuning/prepare_training_data.py data/val_data.pt --samples 100 --split val
```

*   **Dataset Note**: 公開されている `nvidia/PhysicalAI-Autonomous-Vehicles` は、Alpamayo R1 の "Evaluation Dataset" としてModel Cardに記載されています（学習は非公開の80k時間データ）。
    > **Training Dataset:**
    > - Image Training Data Size: More than 1 Billion Images (from **80,000 hours** of multi-camera driving data)
    > - Non-Audio, Image, Text Training Data Size: Trajectory data: **80,000 hours** at 10Hz sampling rate
    >
    > **Quantitative Evaluation Benchmarks:**
    > - Open-Loop Evaluation on the [PhysicalAI-AV Dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles): minADE_6 at 6.4s of 0.85m.


*   **Splittingの仕組み**:
    *   **「100個に1個」のような定期的 (Index) な分割ではありません。**
    *   各データのClip IDをハッシュ化して 0〜99 の値を算出し、その値が **0〜89ならTrain**、**90〜99ならVal** と振り分けています（擬似的なランダム分割）。
    *   これにより、特定の順番に依存せず、データ全体から公平に90%と10%を抽出しています。
    *   `--split train` と `--split val` は互いに排他なので、学習と評価のデータが混ざることはありません。
*   **処理内容**:
    *   指定した Split に属するデータを抽出し、.pt ファイルを作成します。

#### Step 2: 学習の実行 (Training)
Trajectory Decoder (Diffusion Head) のみを再学習させます。
作成した `scripts/finetuning/train_trajectory_decoder.py` を使用します。

```bash
# 使用法: python scripts/finetuning/train_trajectory_decoder.py --data_path <pt_file>
python scripts/finetuning/train_trajectory_decoder.py \
    --data_path data/training_data.pt \
    --output_dir ./checkpoints \
    --epochs 10 \
    --batch_size 4
```

*   **学習の仕組み**:
    *   **VLM Backbone (Freeze)**: 重みを固定し、画像と言語の理解能力（脳）はそのまま維持します。
    *   **Diffusion Head (Train)**: ここだけを学習対象とします。
    *   **Objective**: Flow Matching Loss を最小化し、単眼画像の特徴から正解軌道を生成できるように矯正します。

#### Step 3: 評価の実行 (Evaluation)
学習した重みをロードし、未学習のテストデータ（カーブ等）に対して推論を行い、直進バイアスが解消されたかを確認します。
新たに作成した `scripts/finetuning/evaluate_trajectory_decoder.py` を使用します。

```bash
# 使用法: python scripts/finetuning/evaluate_trajectory_decoder.py --data_path <val_pt_file> --ckpt_path <ckpt_file>
python scripts/finetuning/evaluate_trajectory_decoder.py \
    --data_path data/val_data.pt \
    --ckpt_path ./checkpoints/ckpt_epoch_9.pt \
    --output_dir ./eval_results \
    --num_samples 10
```

1.  **評価の仕組み**:
    *   スクリプト内でベースモデル (`nvidia/Alpamayo-R1-10B`) をロードします。
    *   指定されたチェックポイントから、**Diffusion Headの重みのみ** を上書きロードします。
    *   Valデータに対して推論（軌道生成）を実行し、正解軌道との比較プロットを `eval_results/` に保存します。

2.  **定性評価 (Visualization)**:
    *   `eval_results/*.png` を確認し、単眼入力のみで白線に沿ったカーブ軌道が生成されているかチェックします。
    *   従来の「直進し続ける」挙動が改善されているかを視覚的に判断します。

この手順により、モデルは「単眼画像から得られる視覚情報（白線の曲がり具合など）」に強く依存して軌道を生成するように矯正され、情報の欠損による「とりあえず直進」というバイアスが解消されます。

#### Q. Trajectory Decoder の追加学習で、単眼でも適切な軌道を出力できるか？
