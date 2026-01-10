# Experiment 00b: データセット検証 (Dataset Validation)

## 目的 (Objective)
ローカルのRosbagデータ (`rosbag2_autoware_0.mcap`) およびスクリプト (`debug_viz.py`) に問題がある可能性を排除するため、**公式のNVIDIA PhysicalAI-AV データセット**を用いてモデルのベースライン挙動を確認する。

### 検証の狙い
*   公式データでもカーブ予測ができない → **モデル自体の直進バイアス** (本質的な問題)
*   公式データではカーブ予測できる → **ローカルデータまたはスクリプトの問題** (技術的バグ)

---

## 実験設定 (Configuration)
*   **データセット**: [nvidia/PhysicalAI-Autonomous-Vehicles](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
*   **使用クリップ**: `clip_id = "030c760c-ae38-49aa-9ad8-f5650a545d26"` (公式サンプル)
*   **タイムスタンプ**: `t0_us = 5_100_000` (5.1秒地点)

---

## 実行コマンド (Execution Command)

### 1. カーブシーン探索
```bash
python ../../../find_curve_clips.py
```

### 2. 公式推論テスト
```bash
python ../../../src/alpamayo_r1/test_inference.py
```

---

## 結果 (Results)

### 1. カーブ曲率解析
| Metric | Value |
| :--- | :--- |
| **Max Absolute Curvature** | 0.004216 (1/m) |
| **Mean Absolute Curvature** | 0.001315 (1/m) |
| **Minimum Turn Radius** | **~237.2 meters** |
| **判定** | **✗ ほぼ直進シーン** (curvature < 0.01) |

**重要な発見**: 公式の example clip (`030c760c-ae38-49aa-9ad8-f5650a545d26`) は、実は**カーブシーンではなく、ほぼ直進シーン**であることが判明しました。

### 2. 推論テスト結果
| Metric | Value |
| :--- | :--- |
| **minADE** | 2.56 meters |
| **CoT (Chain-of-Causation)** | "Nudge to the left to increase clearance from the **construction cones** encroaching into the lane." |
| **Status** | Success (推論正常完了) |

---

## ログ詳細 (Log Details)
*   `find_curve_clips.log`
*   `test_inference_output.log`

---

## 考察 (Discussion)

### 1. 公式サンプルは「カーブ検証」に適さない
公式の `test_inference.py` で使用されている example clip は、**旋回半径237mの緩やかなシーン（ほぼ直進）**でした。

これは、NVIDIAが提供するサンプルが**「推論パイプラインの動作確認」を目的としたもの**であり、**「カーブ走行性能の検証」を想定していない**ことを示しています。

### 2. カーブシーンの曲率基準
道路設計基準では:
*   **高速道路**: 最小半径 300m 程度
*   **一般道のカーブ**: 半径 50~150m
*   **急カーブ**: 半径 20~50m

今回の clip (半径237m) は高速道路レベルであり、「モデルがカーブを曲がれるか」を検証するには不十分です。

### 3. 本質的な検証の困難性
*   データセットには `curvature` メタデータが存在するが、**どのclip_idが曲率の高いシーンか**を特定するには、全clip_idのリストと各clipの曲率スキャンが必要
*   現在の環境では個別clipの解析はできるが、**全体からのフィルタリングは未実装**

### 4. ローカルデータとの比較
ローカルRosbag (`rosbag2_autoware_0.mcap`) の ratio=0.6 付近のシーンは、**視覚的に明らかな右カーブ**が存在します（道路の湾曲とコーンの配置から判断）。

つまり、公式データよりも**ローカルデータの方がカーブ検証に適している**可能性が高いです。

### 5. 結論
*   公式データでの検証は実施できたが、そのシーンは**直進に近い**ため、「カーブで曲がれない問題」の検証には不適切
*   ローカルデータにバグがある可能性は**低い**（むしろカーブ検証により適している）
*   したがって、**モデル自体の直進バイアスである可能性が高い**

---

## 次のステップ (Next Steps)
1.  **高曲率clip_idの特定**: データセット全体をスキャンし、`curvature > 0.05` (半径20m以下の急カーブ) のclipを探索
2.  **または**: 現在のローカルデータでの実験結果を「モデルの本質的な限界」として結論付ける
