# 検証レポート: Cosmos Cookbook ファインチューニング適用可能性 (Finetuning Feasibility)

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
    *   直進バイアスの主因である Trajectory Decoder は **Diffusion Model** です。
    *   Cosmos Cookbook には、**Diffusion Loss (MSE / Flow Matching Loss) を計算する機能も、Diffusion Head の重みを更新する機能もありません。**

## 5. 結果のまとめ (Results Summary)

検証の結果、Cosmos Cookbook の適用可能性は以下の通りとなりました。

| 対象モジュール | 役割 | Cosmos Cookbook で学習可能？ | 理由 |
| :--- | :--- | :--- | :--- |
| **Cosmos-Reason Backbone** | 画像認識・思考・CoT記述 | **Yes** (SFT Recipe) | VLMのテキスト生成タスクであるため。 |
| **Trajectory Decoder** | **実走行軌道の生成 (今回の課題)** | **No** | **拡散モデルであり、Cookbookの学習範疇外のため。** |

### Conclusion
**Cosmos Cookbook は使えません。**
今回の「カメラ構成変更による直進バイアス」は、Trajectory Decoder (Diffusion Head) が特定の入力特徴量（Camera ID等）に過剰適合していることが主因と考えられます。
この Head を再学習させるには、Cookbook ではなく、**Trajectory Loss を扱える Alpamayo-R1 独自の学習コード（DeepSpeed実装）** を使用する必要があります。

---

## (Appendix) おまけ: Cosmos Cookbook でファインチューニングを実行する

「使えない」という結論だけではイメージが湧きづらいため、もし「脳 (Brain)」だけを鍛え直すとしたらどのようなコマンドになるのか、Cookbook の **Post-Training (SFT)** レシピを例に示します。

#### 1. 実行レシピ
`recipes/post_training/reason2/video_caption_vqa` (Uberのデータセットを使ったAVキャプション生成の追加学習)

#### 2. 実行コマンド (SFT)
Cookbook では `cosmos-rl` というコマンドラインツールを使って学習を実行します。

```bash
# SFTの実行例 (Cosmos Reason 2)
# 設定ファイル(toml)を指定して学習を開始します
cosmos-rl --config config/uber_sft_blended.toml scripts/uber_dataloader.py
```

これだけで、`uber_sft_blended.toml` に定義されたパラメータ（学習率、エポック数、LoRA設定など）に従って、QwenベースのVLMの再学習が始まります。

#### 3. なぜこれで Trajectory が直らないのか？
この学習を実行すると、VLMは「Uber風の状況説明」を学習します。しかし、**「ハンドルを何度切るか（＝Trajectory Decoderの重み）」を更新するプロセスがこのコマンドには含まれていません。**
そのため、いくらこのコマンドでVLMを賢くしても、最終的な軌道出力のバイアスは解消されないのです。
