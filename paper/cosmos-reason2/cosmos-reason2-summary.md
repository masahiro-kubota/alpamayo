# Cosmos-Reason2 サマリー

> **注意**: 公式論文（arXiv等）は存在しません。本ドキュメントはHugging Faceモデルカード、GitHubリポジトリ、READMEから情報を収集して作成しました。

## 概要

**NVIDIA Cosmos Reason 2** は、Physical AI（物理AI）とロボティクス向けのオープンでカスタマイズ可能な推論VLM（Vision Language Model）です。

| 項目 | 内容 |
|------|------|
| **リリース日** | 2025年12月19日 |
| **開発元** | NVIDIA |
| **ベースモデル** | Qwen3-VL |
| **ライセンス** | Apache 2.0（コード）/ NVIDIA Open Model License（モデル） |

## モデルバリエーション

| モデル | ベースモデル | パラメータ数 | GPU メモリ要件 |
|--------|-------------|-------------|---------------|
| [Cosmos-Reason2-2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B) | [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) | 2,438,696,960 | 24GB |
| [Cosmos-Reason2-8B](https://huggingface.co/nvidia/Cosmos-Reason2-8B) | [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | 8,767,123,696 | 32GB |

## Cosmos-Reason1 → Reason2 の主な改善点

| 機能 | Reason1 | Reason2 |
|------|---------|---------|
| ベースモデル | （不明、おそらくQwen2.5系） | **Qwen3-VL** |
| 時空間理解 | ✓ | **強化**（タイムスタンプ精度向上） |
| 物体検出 | ✓ | **2D/3D座標＋バウンディングボックス** |
| コンテキスト長 | （不明） | **最大256Kトークン** |

## アーキテクチャ

```
入力 → Vision Transformer (ViT) → Dense Transformer (LLM) → 出力
       [Vision Encoder]           [Qwen3-VL-8B-Instruct]
```

- **入力**: テキスト + 動画/画像
  - 推奨FPS: 4（学習時の設定に合わせる）
  - 対応形式: mp4（動画）、jpg（画像）
- **出力**: テキスト（推論トレース + 回答）
  - 推奨max_tokens: 4096以上（長いCoT応答の切り詰め防止）

## 主なユースケース

### 1. ビデオ分析AIエージェント
大量のビデオデータからインサイトを抽出し、根本原因分析を実行。都市や産業運用における録画/ライブストリームの分析に活用。

### 2. データキュレーション・アノテーション
大規模で多様なトレーニングデータセットの高品質なキュレーションとアノテーションを自動化。NVIDIA Cosmos Curatorと連携。

### 3. ロボット計画・推論
ロボットVLA（Vision-Language-Action）モデルの「頭脳」として機能。ヒューマノイドや自動運転車が環境を解釈し、タスクを分解・実行可能に。

## サポートするプロンプトテンプレート

### Captioning（キャプショニング）
- Caption: 動画/画像の説明生成
- Temporal Localization: 時間軸でのイベント特定
- Describe Anything: 任意の対象の詳細説明
- 2D Grounding: バウンディングボックス座標の取得

### Embodied Reasoning（具現化推論）
```yaml
# 例: embodied_reasoning.yaml
user_prompt: |
  What can be the next immediate action?
```

```yaml
# 例: av_cot.yaml（自動運転向け）
user_prompt: |
  The video depicts the observation from the vehicle's camera. 
  You need to think step by step and identify the objects in the scene 
  that are critical for safe navigation.
```

```yaml
# 例: robot_cot.yaml（ロボット向け）
user_prompt: |
  You are given the task "{task_instruction}". 
  Specify the 2D trajectory your end effector should follow in pixel space.
  Return the trajectory coordinates in JSON format.
```

## 推論速度（参考値）

サンプルログ（`temporal_localization.log`）より：

| 指標 | 速度 |
|------|------|
| 入力トークン | ~1,975 tok/s |
| 出力トークン | ~51 tok/s |

※ GPU・バッチサイズ・コンテキスト長により変動

## Post-Training（ファインチューニング）

### 方法1: TRL（Transformers Reinforcement Learning）
- **SFT**: QLoRAを使用したSupervised Fine-Tuning
- **GRPO**: Group Relative Policy Optimization

Google Colabで実行可能なノートブック提供：
- `trl_sft.ipynb`
- `trl_grpo.ipynb`

### 方法2: Cosmos-RL
NVIDIAの非同期Post-Trainingフレームワーク。SFTとRLHFに特化。

**最小要件**: 4 GPU × 80GB メモリ

```bash
# SFT実行例
uv run cosmos-rl --config configs/llava_sft.toml --log-dir outputs/llava_sft scripts/llava_sft.py
```

## 量子化

[llmcompressor](https://github.com/vllm-project/llm-compressor)を使用：

```bash
./scripts/quantize.py -o /tmp/cosmos-reason2/checkpoints
# オプション: --precision fp4
```

## 推論実行

### vLLMでのオンラインサービング

```bash
vllm serve nvidia/Cosmos-Reason2-2B \
  --allowed-local-media-path "$(pwd)" \
  --max-model-len 16384 \
  --media-io-kwargs '{"video": {"num_frames": -1}}' \
  --reasoning-parser qwen3 \
  --port 8000
```

### Transformersでの推論

```python
import transformers
import torch

model_name = "nvidia/Cosmos-Reason2-2B"
model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
    model_name, dtype=torch.float16, device_map="auto", attn_implementation="sdpa"
)
processor = transformers.Qwen3VLProcessor.from_pretrained(model_name)

conversation = [
    {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
    {"role": "user", "content": [
        {"type": "video", "video": "sample.mp4"},
        {"type": "text", "text": "Caption the video in detail."},
    ]},
]

inputs = processor.apply_chat_template(
    conversation, tokenize=True, add_generation_prompt=True,
    return_dict=True, return_tensors="pt", fps=4
)
generated_ids = model.generate(**inputs, max_new_tokens=4096)
```

## テスト済みハードウェア

| GPU | CUDA | 対応機能 |
|-----|------|----------|
| NVIDIA H100 | 12.8 | 推論/学習/量子化 |
| NVIDIA GB200 | 13.0 | 推論 |
| NVIDIA DGX Spark | 13.0 | 推論 |
| NVIDIA Jetson AGX Thor | 13.0 | Transformers推論（vLLM近日対応） |

## 学習データセット

Cosmos-Reason1のデータセットに加え、以下を追加：

- EgoExo4D
- PerceptionTest
- Language Table
- IntPhys
- InfLevel
- CLEVRER

## 関連リソース

- [Hugging Face Collection](https://huggingface.co/collections/nvidia/cosmos-reason2)
- [GitHub Repository](https://github.com/nvidia-cosmos/cosmos-reason2)
- [Cosmos Cookbook](https://nvidia-cosmos.github.io/cosmos-cookbook/index.html)
- [Qwen3-VL Repository](https://github.com/QwenLM/Qwen3-VL)
- [vLLM Documentation](https://docs.vllm.ai/en/stable/)

## Alpamayoとの関係

| モデル | VLMバックボーン |
|--------|----------------|
| **Alpamayo-R1** | Cosmos-Reason**1** |
| Cosmos-Reason2 | Qwen3-VL（別系統） |

Alpamayo-R1はCosmos-Reason1をベースにしており、Cosmos-Reason2とは**異なるアーキテクチャ**です。
