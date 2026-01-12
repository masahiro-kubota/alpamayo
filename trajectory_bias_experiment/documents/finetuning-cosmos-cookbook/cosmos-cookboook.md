### 3. Cosmos RL Framework
Cosmos Cookbook の事後学習 (Post-training) レシピを実行するためには、**Cosmos RL** というフレームワークが必須となります。

*   **概要**: NVIDIA Cosmos モデル専用の強化学習 (RL) および SFT (Supervised Fine-Tuning) 実行エンジン。物理AI向けに設計されており、並列化技術 (Tensor/Sequence/Pipeline Parallelism) を駆使して大規模モデルの効率的な学習をサポートします。
*   **役割**: Cookbook が「手順書（レシピ）」であるのに対し、Cosmos RL は実際に計算リソースを使って学習プロセスを回す「実行ツール（調理器具）」です。
*   **構成**: Policy（学習対象モデル）と Rollout（推論/データ生成）を分離した非同期アーキテクチャを採用しており、大規模な強化学習ループを効率的に回せる設計になっています。

本レポートで検証しているファインチューニング手順も、この `cosmos-rl` コマンドを使用して実行される想定です。

### 4. SFT (Supervised Fine-Tuning) とは
**SFT (教師ありファインチューニング)** とは、事前学習済みのモデル（ここでは Cosmos Reason）に対して、特定のタスクに適応した「入力と正解（教師データ）」のペアを追加学習させるプロセスです。

*   **目的**: 一般的な知識しか持たないモデルに、特定のドメイン（今回は Uber の自動運転）特有の言葉遣い、注目すべきポイント、出力フォーマットを教え込むこと。
*   **方法**: 「動画 + プロンプト」を入力し、人間が作成した「理想的なキャプション」を正解として与え、その誤差を埋めるようにモデルの重みを微調整します。
*   **効果**: SFTを行うことで、モデルは「空が青い」といった一般的な描写だけでなく、「自車は直進中で、前方の交差点信号は赤である」といった、自動運転開発に必要な具体的かつ正確な記述ができるようになります。

### Conclusion
**Cosmos Cookbook は使えません。**
今回の「カメラ構成変更による直進バイアス」は、Trajectory Decoder (Diffusion Head) が特定の入力特徴量（Camera ID等）に過剰適合していることが主因と考えられます。
この Head を再学習させるには、Cookbook ではなく、**Trajectory Loss を扱える Alpamayo-R1 独自の学習コード（DeepSpeed実装）** を使用する必要があります。

---

## (Appendix) おまけ: Cosmos Cookbook でファインチューニングを実行する

「使えない」という結論だけではイメージが湧きづらいため、もし「脳 (Brain)」だけを鍛え直すとしたらどのようなコマンドになるのか、Cookbook の **Post-Training (SFT)** レシピを例に示します。

#### 1. 実行レシピ
`recipes/post_training/reason2/video_caption_vqa` (Uberのデータセットを使ったAVキャプション生成の追加学習)

#### 2. 学習の実行 (Training)
Cookbook では `cosmos-rl` というコマンドラインツールを使って学習を実行します。

```bash
# SFTの実行例 (Cosmos Reason 2)
# 設定ファイル(toml)を指定して学習を開始します
cosmos-rl --config config/uber_sft_blended.toml scripts/uber_dataloader.py
```

**必要なファイルの所在確認結果:**
*   **Dataloader Script**: `scripts/examples/reason1/av_video_caption_vqa/scripts/avha_sft.py`
    *   ドキュメントの `scripts/uber_dataloader.py` に相当する実装が別名で見つかりました。Uberデータセットのファイル名形式にハードコードで対応しており、これをリネームして利用可能です。
*   **Config File**: ドキュメント内のコードブロック (`uber_sft_blended.toml`) から作成可能。

これらを用いれば、Cosmos RL を介して学習プロセスを再現可能です。

#### 3. 評価の実行 (Evaluation)
学習コマンド (`cosmos-rl`) は **学習のみ** を行います。評価（推論とスコア算出）には別のスクリプトを使用します。
ドキュメント上のファイル名と実態が異なりますが、`scripts/examples/reason1/av_video_caption_vqa/` にあるスクリプトが Reason 2 (Qwen3) にも対応しています。

**Step 1: 推論実行 (Inference)**
`avha_caption.py` を使用します。
```bash
python scripts/examples/reason1/av_video_caption_vqa/avha_caption.py \
    --model-path outputs/uber/ckpt-250 \
    --video-dir ./eval/videos \
    --output-dir ./output/reason2_sft
```

**Step 2: スコア算出 (Scoring)**
`avha_judge.py` (LLM-as-a-Judge) を使用します。
```bash
python scripts/examples/reason1/av_video_caption_vqa/avha_judge.py \
    --output-dir ./output/reason2_sft \
    --answer-dir ./eval/metas \
    --score-dir ./scores/reason2_sft
```

#### 4. GPUなし環境での検証可能性 (Feasibility without GPU)
強力なGPU（A100/H100等）がない場合、どこまで検証可能かを整理しました。

| Step | 処理内容 | GPUなしでの検証可否 | 理由 |
| :--- | :--- | :--- | :--- |
| **0. Data Prep** | データのダウンロード・配置・前処理 | **可能 (Yes)** | 公開データのダウンロードとファイル配置、JSONの確認は可能。 |
| **1. Setup** | 環境構築 (pip install) | **可能 (Yes)** | ライブラリのインストールは可能。 |
| **2. Dry Run** | コード動作確認 (Dryrun) | **可能 (Yes)** | `avha_caption.py` 等には `--dryrun` オプションがあり、モデルロードをスキップしてロジックの疎通確認・パス確認が可能。 |
| **3. Training** | 実際の学習 (SFT) | **不可 (No)** | 数千ステップのBackpropagationが必要。CPUのみでは実行不可。 |
| **4. Inference** | 推論 (Caption生成) | **不可 (No)** | 7B/8BクラスのVLMを動画入力で動かすにはVRAMが必須。 |
| **5. Scoring** | スコア算出 (LLM Judge) | **可能 (Yes)** | `avha_judge.py` は OpenAI API (GPT-4) を叩くだけなので、推論結果のJSONさえあれば手元のPCでも実行可能。 |

**結論**: コードの動作検証（Dry Run）とデータパイプラインの確認までは、GPUなしでも進められます。

#### 5. 期待される改善効果 (Evaluation Results)
Uberデータセットを用いたこのレシピでは、ベースモデル (Cosmos Reason 2-8B) と比較して以下の指標改善が報告されています。

- **BLEU Score**: **+10.6%** (0.113 -> 0.125)
    - キャプションの語彙的正確性が向上。
- **MCQ-based VQA**: **+0.67pt** (80.18% -> 80.85%)
    - 映像内容に対する質問応答精度が向上。
- **LingoQA**: **+13.8pt** (63.2% -> 77.0%)
    - **特に外部ベンチマークでの改善が著しく、自動運転ドメインの汎化性能が向上しています。**

#### 4. なぜこれで Trajectory が直らないのか？
この学習を実行すると、VLMは「Uber風の状況説明」を学習します。しかし、**「ハンドルを何度切るか（＝Trajectory Decoderの重み）」を更新するプロセスがこのコマンドには含まれていません。**
そのため、いくらこのコマンドでVLMを賢くしても、最終的な軌道出力のバイアスは解消されないのです。
