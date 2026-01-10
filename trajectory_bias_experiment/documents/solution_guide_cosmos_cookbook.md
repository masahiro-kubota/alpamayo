# ソリューションガイド: NVIDIA Cosmos Cookbookを用いた直進バイアスの解消

本ガイドでは、**NVIDIA Cosmos Cookbook** を活用して、Alpamayo-R1に見られる「直進バイアス」および「思考と行動の乖離（Core Alignment Problem）」を解消するための具体的なワークフローを解説します。

## 1. 概要 (Overview)
公式リポジトリ (`NVlabs/alpamayo`) は推論専用ですが、**Cosmos Cookbook** (`nvidia-cosmos/cosmos-cookbook`) には、モデルをカスタマイズ（ファインチューニング）するためのレシピが用意されています。これを利用して、**日本の道路やカーブデータに適応したモデル**を再構築します。

*   **Target Repository**: `https://github.com/nvidia-cosmos/cosmos-cookbook`
*   **Goal**: カーブ走行時のデータを追加学習し、Reasoning（カーブ認識）とAction（旋回軌道）の一貫性を獲得させる。

## 2. なぜ "Cosmos" なのか？ (Relationship between Alpamayo and Cosmos)
ユーザーが疑問に思う点として、「Alpamayoのファインチューニングになぜ Cosmos が出てくるのか？」が挙げられます。関係性は以下の通りです。

1.  **親子関係**: Alpamayo-R1 は、NVIDIAの汎用基盤モデル **"Cosmos Reason"** をベースに構築された、自律走行（AV）特化モデルです。
2.  **エコシステム**: NVIDIAは、Alpamayoを含む全てのCosmos系モデル共通の開発ツールチェーンとして **"Cosmos Cookbook"** を提供しています。
3.  **結論**: したがって、Alpamayo独自の独立した学習ツールがあるわけではなく、**親元であるCosmosの標準ツール（Cookbook）を使うのが正規の手順**となります。

---

## 3. 実施ステップ (Workflow)

### Step 1: 環境構築 (Setup)
Cosmos Cookbookを作業環境にクローンし、必要な依存関係をインストールします。

```bash
git clone https://github.com/nvidia-cosmos/cosmos-cookbook.git
cd cosmos-cookbook
pip install -r requirements.txt
```

#### 推奨ハードウェア要件 (Hardware Requirements)
モデルサイズ (10B parameters) に基づく目安です。

| 手法 | 必要VRAM | 推奨GPU構成 | 備考 |
|---|---|---|---|
| **Full Fine-tuning** | 80GB+ | **A100 (80GB) x 1~2** or **H100** | 全パラメータ更新。商用レベルの精度を求める場合。 |
| **LoRA** | 24GB+ | **RTX 3090 / 4090 (24GB)** | 学習パラメータを削減し、コンシューマハイエンドGPUで実行可能。 |
| **QLoRA (4-bit)** | 16GB~ | **RTX 4080 / 3060 (12GB+)** | 量子化によりさらに軽量化。試行錯誤に最適。 |

> **推奨**: まずは計算コストの低い **LoRA (24GB VRAM環境)** での検証を推奨します。Cosmos Cookbookの標準レシピもLoRAをサポートしています。

### Step 2: データセット作成 (Data Curation)
直進バイアスを解消するには、**「カーブを曲がっている成功事例」**のみを集めた高品質なデータセットが必要です。
Alpamayo-R1の学習形式（3要素ペア）に合わせてデータを整形します。

#### 必要なデータ構造 (JSONL形式等の例)
各サンプルは以下の3要素を持つ必要があります。
1.  **Vision (画像)**: カーブに差し掛かっているフロント/サイドカメラ画像。
2.  **Reasoning (思考/CoT)**: 「前方右カーブあり。車線を維持するために右へ操舵し、減速する」といったテキスト。
3.  **Action (軌道)**: 上記の思考に対応する、**実際に曲がっている**将来軌道 (x, y, yaw)。

> **Point**: ここで「思考はカーブだが、軌道は直進」という矛盾データを含めないことが最重要です。

#### 単眼（フロント）カメラのみでの学習について (Single Camera Support)
コード (`src/alpamayo_r1/helper.py`) の解析により、本モデルは入力画像の枚数を動的に処理する設計であることが確認されました（`[{"image": frame} for frame in frames]`）。
したがって、**フロントカメラ画像のみを用いたファインチューニングも技術的に可能**です。
*   **メリット**: データ収集・作成コストが1/3に削減できる。
*   **注意点**: 死角が増えるため、車線変更や複雑な交差点などのタスク性能が変化する可能性がありますが、「直進バイアス解消（カーブ追従）」という目的においては単眼でも十分な効果が期待できます。

### Step 3: ファインチューニング (Post-training)
Cookbook内の **LoRA (Low-Rank Adaptation)** レシピを使用します。フルパラメータチューニングと比較して、少ない計算リソースで効率的に挙動を矯正できます。

*   **対象レシピ**: `recipes/post_training/lora_finetune.py` (※パスは最新のRepo構成に準拠)
*   **設定のポイント**:
    *   **Target Modules**: 言語モデル(LLM)部分だけでなく、**Action Head (Diffusion Decoder)** にもLoRAアダプタを適用すること。これにより、思考結果を軌道生成に反映させる接続部分を強化します。
    *   **Loss Function**: CoT生成のLossに加え、Trajectory予測のMSE Loss等を複合的に最小化する設定を確認します。

### Step 4: 評価 (Evaluation)
学習後、Cookbookに含まれる評価スクリプトを用いてバイアスの解消を確認します。

1.  **Open-Loop Evaluation**: テスト用カーブデータセットに対し、生成された軌道と正解軌道の誤差（ADE/FDE）を計測。
2.  **Consistency Check**: カーブシーンにおいて、「CoTがカーブを認識しているか」かつ「軌道がカーブしているか」の一致率を確認。

---

## 4. 期待される効果
このプロセスを経ることで、Alpamayo-R1は以下のように改善される見込みです。
*   **Before**: 「右カーブだ（思考）」→ 直進する（行動） [不整合]
*   **After**: 「右カーブだ（思考）」→ **右に曲がる（行動）** [一致]

## 5. 参考リンク
*   [NVIDIA Cosmos Cookbook GitHub](https://github.com/nvidia-cosmos/cosmos-cookbook)
*   [Alpamayo-R1 Model Card (Hugging Face)](https://huggingface.co/nvidia/Alpamayo-R1-10B)
