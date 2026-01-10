# Cosmos Cookbookによる単眼カメラ向けファインチューニング挑戦記

## 0. TL;DR
- **What**: 自動運転VLAモデル **Alpamayo-R1** を単眼カメラ（フロントのみ）で使用すると、「思考はカーブと認識しているのに直進する」という致命的なバイアスが発生する。この問題を **NVIDIA Cosmos Cookbook** によるファインチューニングで解決することを目指す。
- **Result**: （実験進行中）
- **So What**: ファインチューニングにより、モデルの脳構造を「4眼の立体視依存」から「単眼の消失点依存」へ適応させ、単眼カメラのみでもカーブを曲がれるようにする。

---

## 1. 目的（問題意識・課題感）
- **What**: [カメラ構成実験](camera_experiments/camera_configuration_experiment.md) の結果、Alpamayo-R1 が単眼カメラ入力では**カーブを認識しているにも関わらず直進し続ける**ことが判明した。
- **Why**: 
    - 学習済みモデルは「4カメラ等の複数視点からの幾何学的情報（視差など）」に依存して位置推定を行っている。
    - 情報量が減った単眼では自信を持てず、無難な「直進」を出力してしまっている状態。
    - 自動運転AIチャレンジのシミュレータ環境には**フロントカメラしかない**ため、このままでは使えない。
- **Goal**: ファインチューニングにより、**単眼でもカーブを曲がれるモデル**を作成する。

## 2. Approach

### なぜファインチューニングで解決できるのか？

現状のモデルが単眼入力でカーブを曲がれないのは、**「単眼の画角だけでカーブの深さを推定する」という重み付けがされていない**ためです。

**ファインチューニングの狙い**:
- モデルの脳構造を **「4眼の立体視依存」** から **「単眼の消失点依存」** へ適応（矯正）させる
- 視差情報がなくても、道路の消失点や車線のカーブ具合から旋回度合いを推定できるようにする

**重要なポイント**: 
- 現在のモデルが出力してしまう「思考はカーブ、軌道は直進」という**矛盾したデータを含めない**こと
- **「思考も軌道もカーブしている正解データ」**だけを、入力画像を単眼にした状態で再学習させるのが鍵

これにより、モデルは「フロント画像1枚でこの景色が見えたら、思考だけでなく身体（軌道）も曲げる」という新しい相関関係を学習します。

### なぜ Cosmos Cookbook なのか？

公式リポジトリ `NVlabs/alpamayo` は**推論専用**です。学習（ファインチューニング）を行うには、**Cosmos Cookbook** を使用します。

**Alpamayo-R1** と **Cosmos** の関係：
1. **親子関係**: Alpamayo-R1 は、NVIDIAの汎用基盤モデル **"Cosmos Reason"** をベースに構築された、自律走行（AV）特化モデル
2. **エコシステム**: NVIDIAは、Alpamayo を含む全ての Cosmos系モデル共通の開発ツールチェーンとして **"Cosmos Cookbook"** を提供
3. **結論**: Alpamayo独自の学習ツールがあるわけではなく、**親元である Cosmos の標準ツール（Cookbook）を使うのが正規の手順**

## Theory: ファインチューニングの学習メカニズム

### 単眼カメラのみでの学習が可能な理由

コード (`src/alpamayo_r1/helper.py`) の解析により、本モデルは入力画像の枚数を動的に処理する設計であることが確認されています（`[{"image": frame} for frame in frames]`）。

したがって、**フロントカメラ画像のみを用いたファインチューニングも技術的に可能**です。

- **メリット**: データ収集・作成コストが1/3に削減できる
- **注意点**: 死角が増えるため、車線変更や複雑な交差点などのタスク性能が変化する可能性がある。ただし「直進バイアス解消（カーブ追従）」という目的においては単眼でも十分な効果が期待できる

### LoRA による効率的な適応

フルパラメータチューニングは 80GB+ の VRAM を必要とするため、**LoRA (Low-Rank Adaptation)** を採用します。

**設定のポイント**:
- **Target Modules**: 言語モデル（LLM）部分だけでなく、**Action Head（Diffusion Decoder）** にもLoRAアダプタを適用すること。思考→軌道の接続部分を強化する。
- **Loss Function**: CoT生成の Loss に加え、Trajectory 予測の MSE Loss を複合的に最小化する。

## 3. 前提環境 (Prerequisites)
- **Base Model**: Alpamayo-R1 (10B Parameters)
- **Training Framework**: [NVIDIA Cosmos Cookbook](https://github.com/nvidia-cosmos/cosmos-cookbook)
- **Training Recipe**: `recipes/post_training/lora_finetune.py`

#### ハードウェア要件

| 手法 | 必要VRAM | 推奨GPU | 備考 |
|---|---|---|---|
| **Full Fine-tuning** | 80GB+ | **A100 (80GB) x 1~2** or **H100** | 全パラメータ更新。商用レベルの精度を求める場合。 |
| **LoRA** | 24GB+ | **RTX 3090 / 4090 (24GB)** | 学習パラメータを削減し、コンシューマハイエンドGPUで実行可能。 |
| **QLoRA (4-bit)** | 16GB~ | **RTX 4080 / 3060 (12GB+)** | 量子化によりさらに軽量化。試行錯誤に最適。 |

> **方針**: まずは計算コストの低い **LoRA (24GB VRAM環境)** での検証を推奨。

## 4. 具体的な検証手順 (Concrete Steps)

### Step 1: 環境構築 (Setup)

```bash
git clone https://github.com/nvidia-cosmos/cosmos-cookbook.git
cd cosmos-cookbook
pip install -r requirements.txt
```

<!-- TODO: 実際のセットアップ時の依存関係問題やバージョン指定をここに追記 -->

### Step 2: データセット作成 (Data Curation)

直進バイアスを解消するには、**「カーブを曲がっている成功事例」**のみを集めた高品質なデータセットが必要です。

#### 必要なデータ構造 (JSONL形式等の例)
各サンプルは以下の3要素を持つ必要があります。
1. **Vision (画像)**: カーブに差し掛かっているフロントカメラ画像（**1枚のみ**）
2. **Reasoning (思考/CoT)**: 「前方右カーブあり。車線を維持するために右へ操舵し、減速する」といったテキスト
3. **Action (軌道)**: 上記の思考に対応する、**実際に曲がっている**将来軌道 (x, y, yaw)

> **Point**: ここで「思考はカーブだが、軌道は直進」という矛盾データを含めないことが最重要です。

<!-- QUESTION: 学習データとして以下のどれを使用予定ですか？
1. AIチャレンジのシミュレータから収集した走行ログ
2. NVIDIA PhysicalAI-AV 公式データセットのカーブシーンを抽出
3. その他（自前で収集など）
-->

### Step 3: ファインチューニング実行 (Post-training)

Cookbook内の **LoRA (Low-Rank Adaptation)** レシピを使用します。

```bash
# 実行コマンド（予定）
python recipes/post_training/lora_finetune.py \
    --model nvidia/Alpamayo-R1-10B \
    --dataset path/to/curve_dataset \
    --lora_rank 16 \
    --lora_alpha 32
```

<!-- TODO: 実際に使用した設定値・コマンドをここに追記 -->

### Step 4: 評価 (Evaluation)

学習後、以下の方法でバイアスの解消を確認します。

1. **Open-Loop Evaluation**: テスト用カーブデータセットに対し、生成された軌道と正解軌道の誤差（ADE/FDE）を計測
2. **Consistency Check**: カーブシーンにおいて、「CoTがカーブを認識しているか」かつ「軌道がカーブしているか」の一致率を確認
3. **Before/After 比較**: 同一のテストケース（f789b390）で、ファインチューニング前後の Max Deviation を比較

## 5. 結果のまとめ (Results Summary)

<!-- TODO: 実験完了後に結果を追記 -->

| 条件 | Max Deviation | 結果 |
|---|---|---|
| **Before (単眼・Padding)** | 3.36 m | Partial（曲がりきれず） |
| **After (ファインチューニング後)** | (TBD) | (TBD) |

## 6. 考察

<!-- TODO: 実験完了後に考察を追記 -->

### 期待される効果
このプロセスを経ることで、Alpamayo-R1は以下のように改善される見込みです。
- **Before**: 「右カーブだ（思考）」→ 直進する（行動） [不整合]
- **After**: 「右カーブだ（思考）」→ **右に曲がる（行動）** [一致]

### 現時点での課題・疑問

1. Cosmos Cookbook の最新バージョンで Alpamayo-R1 のファインチューニングが正式にサポートされているか？
2. 単眼入力でのファインチューニング事例は公開されているか？
3. 必要なデータ量の目安は？（数百サンプルで効果が出るか、数千必要か）

---

## 7. 参考リンク
- [NVIDIA Cosmos Cookbook GitHub](https://github.com/nvidia-cosmos/cosmos-cookbook)
- [Alpamayo-R1 Model Card (Hugging Face)](https://huggingface.co/nvidia/Alpamayo-R1-10B)
- [カメラ構成実験の詳細](camera_experiments/camera_configuration_experiment.md)
- [Cosmos Cookbook ソリューションガイド](solution_guide_cosmos_cookbook.md)
