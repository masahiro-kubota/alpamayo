# Cosmos-RL Codebase Summary

**Cosmos-RL**は、NVIDIA Cosmos モデル群 (Cosmos-Reason, Cosmos-Predict など) の強化学習を行うためのスケーラブルなフレームワークです。
Physical AI (ロボティクス) や World Foundational Models (WFM) への適用を念頭に設計されており、大規模な分散学習を強みとしています。

## 1. 概要と特徴

*   **目的**: LLM, VLM, WFM (世界モデル) などの大規模モデルに対する強化学習 (RLHF/RLAIF) および SFT。
*   **物理AI (Physical AI) への特化**: ロボティクスなどの実世界応用を見据えた設計。
*   **アーキテクチャ**:
    *   **Policy (学習) と Rollout (推論) の完全分離**: 非同期実行による高効率化。
    *   **Single-Controller**: 軽量なコントローラーによる統括。
    *   **6D Parallelism**: Tensor, Sequence, Context, Pipeline, FSDP, DDP のサポート。

## 2. サポートされている手法 (Algorithms)

GRPOとDDRLだけでなく、以下の手法も実装されています。

### A. LLM / VLM (Reasoning Models)
Cosmos-Reason などの学習に使用されます。

*   **SFT (Supervised Fine-Tuning)**: 基本的なファインチューニング機能。
*   **GRPO (Group Relative Policy Optimization)**: メインのRLアルゴリズム。
    *   **Variants**: 設定により以下の派生手法に切り替え可能です (`configs` で指定)。
        *   **DAPO**: GRPOの発展形 (Dataset-Aware Policy Optimization と思われる)。
        *   **GSPO**: 別の派生形。
        *   **AIPO**: Asynchronous Importance weighted Policy Optimization。
    *   **機能**: Off-Policy Masking (DeepSeek-V3.2)、Teacher Distillation など高度な機能を含む。

### B. VLA (Vision-Language-Action Models)
ロボティクス向けモデル (Pi05など) で使用されます。

*   **PPO (Proximal Policy Optimization)**:
    *   `cosmos_rl/policy/trainer/vla_trainer/pi05_trainer.py` 内で、**Standard PPO clip** や **Dual-clip** を用いた実装が確認できます（クラス名は `GRPOTrainer` を継承していますが、中身はPPOロジックを含んでいます）。

### C. WFM (World Foundational Models)
動画生成などの世界モデルで使用されます。

*   **DDRL (Data-regularized RL)**: 拡散モデル向け。Data Loss (Diffusion Loss) を正則化として使用。
*   **FlowGRPO**: Flow Matching モデル向けの GRPO。

## 3. ディレクトリ構成と重要ファイル

| パス | 説明 |
| :--- | :--- |
| `cosmos_rl/dispatcher/algo/grpo.py` | GRPO アルゴリズム定義。 |
| `cosmos_rl/policy/trainer/llm_trainer/grpo_trainer.py` | LLM向け GRPO/DAPO/GSPO 学習ループ。Off-policy masking の実装あり。 |
| `cosmos_rl/policy/trainer/vla_trainer/pi05_trainer.py` | VLA (Pi05) 向けトレーナー。**PPO** のロジックが含まれる。 |
| `cosmos_rl/policy/trainer/wfm_trainer.py` | WFM (動画生成) 向けトレーナー。**DDRL** の実装が含まれる。 |
| `cosmos_rl/policy/trainer/llm_trainer/sft_trainer.py` | SFT 用トレーナー。 |
| `cosmos_rl/policy/config/__init__.py` | 全体の設定定義。`variant` フィールドで GRPO/DAPO/GSPO を切り替える仕様が確認できる。 |

## 4. Alpamayo 開発への活用イメージ

*   **Reasoning 特化**: `grpo_trainer.py` をベースに、GRPO またはその改良版である DAPO/AIPO を試すのが王道です。
*   **ロボティクス/Action**: もし Action 出力を伴う場合は、`pi05_trainer.py` の PPO 実装が参考になります。
*   **世界モデル**: 動画生成要素には `wfm_trainer.py` の DDRL が最適です。

## 5. cosmos-cookbook との関係 (利用イメージ)

**cosmos-rl** はあくまで「ライブラリ/エンジン」であり、ユーザーが直接このリポジトリのコードを修正して学習を回すわけではありません。
実際の学習スクリプト（エントリーポイント）やデータセットの定義は、**`cosmos-cookbook`**（またはそれを模したユーザーリポジトリ）側に実装します。

*   **cosmos-rl**: `pip install` して使うバックエンド機能群。
    *   アルゴリズム (GRPO, PPO, DDRL)
    *   分散学習機構 (Dispatcher, Worker)
    *   報酬サーバー (Reward Service)
*   **cosmos-cookbook**: ユーザーが実際に触る場所。
    *   学習スクリプト (`custom_grpo.py`, `custom_sft.py`)
    *   データローダーの定義
    *   学習設定ファイル (`config.toml`)

Alpamayo の学習を行う際は、`cosmos-cookbook` の `recipes` を参考に自前の学習スクリプトを用意し、そこから `cosmos-rl` の機能を呼び出す形になります。
