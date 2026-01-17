# Cosmos-RL と Cosmos-Cookbook の関係と使用方法

ご認識の通りです。**`cosmos-rl` はライブラリ/ツールとしてインストールし、`cosmos-cookbook` にあるスクリプトや設定ファイルを食わせて実行する** という使い方が基本設計となっています。

## 1. 実行スタイル
`cosmos-rl` コマンドに対し、設定ファイル (`.toml`) と、実行したい Python スクリプトを与えて起動します。

```bash
# SFTの例 (Cosmos-Cookbookのレシピより)
cosmos-rl --config configs/avha_sft.toml scripts/avha_sft.py

# RL (GRPO) の例
cosmos-rl --config configs/videophy2_rl.toml scripts/custom_grpo.py
```

*   **`cosmos-rl`**: 基盤となる学習トレーナー、分散処理、アルゴリズム (GRPO/PPO/DDRL) の実装本体。`pip install` で導入します。
*   **`cosmos-cookbook`**: ユーザーが用意すべき「データの読み込み方 (Dataset)」や「報酬の計算方法 (Reward)」を定義したスクリプト (`custom_sft.py` や `custom_grpo.py`) と、ハイパーパラメータ設定 (`.toml`) の集合です。

## 2. 実装されているレシピ (Cookbook側)

Cookbook 内を調査した結果、以下のスクリプトが確認できました。

*   **SFT (Supervised Fine-Tuning)**: 充実しています。
    *   `scripts/examples/reason1/av_video_caption_vqa/scripts/avha_sft.py` (自動運転動画のキャプション生成)
    *   `scripts/examples/reason1/intelligent-transportation/custom_sft.py`
*   **GRPO (Reinforcement Learning)**: 実装例があります。
    *   `scripts/examples/reason1/physical-plausibility-check/custom_grpo.py` (物理法則の妥当性判定)
*   **PPO (Robotics/VLA)**:
    *   `cosmos-rl` **本体**には `pi05_trainer.py` として PPO のロジックが存在しますが、**Cookbook 内には PPO を使う具体的なレシピ（スクリプト）は現状見当たりません**でした。
    *   VLA (Vision-Language-Action) 向けの機能であるため、まだ公開されているレシピが少ないか、内部向け限定の可能性があります。

**結論**:
Alpamayo の開発においては、Cookbook にある **「SFT」と「GRPO」のスクリプトをテンプレートとして利用し、Alpamayo 用のデータや報酬関数に書き換えて `cosmos-rl` コマンドで実行する** 形が最短ルートになります。
