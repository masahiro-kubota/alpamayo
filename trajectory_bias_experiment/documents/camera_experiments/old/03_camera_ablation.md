# Experiment 02: Camera Ablation (カメラ構成検証)

## 1. 目的（問題意識・課題感）
- **What**: 標準の4カメラ構成から特定のカメラを排除（Ablation）し、カーブ走行における各カメラの寄与度を明らかにしたい。
- **Why**:
    - 「直進バイアス」の原因が、単純なデータ不足なのか、特定の視覚情報（サイドビュー）の欠如によるものなのかを切り分ける必要がある。
    - 特に、高価なTeleカメラやサイドカメラが実際に推論（Trajectory生成）にどれだけ寄与しているかを確認したい。
- **Goal**: サイドカメラ（Left/Right）およびTeleカメラの有無が、急カーブ（Clip `f789b390`）の旋回予測に与える影響を定量的に検証する。

## 2. 実験結果 (Results)
本実験では、以下の4つの条件で推論を行い、その結果を比較した。
目標軌道との最大横方向偏差（Max Lateral Deviation）が大きいほど、適切にカーブを認識して曲がれていることを示す。

### 結果一覧 (Summary Table)
| ID | 条件 (Condition) | 入力形式 (Input Type) | カメラ構成 (Cameras) | Max Dev | 結果 (Result) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **Teleなし** | Variable Length | Left, Front, Right | **6.45 m** | **Success** (Curved) |
| **2** | **Teleなし・黒埋め** | Padding (Black) | Left, Front, Right, *Black* | **9.50 m** | **Success** (Perfect) |
| **3** | **フロントのみ** | Variable Length | Front, Tele | **0.14 m** | **Failure** (Straight) |
| **4** | **フロントのみ・黒埋め** | Padding (Black) | *Black*, Front, *Black*, Tele | **3.36 m** | **Partial** (Understeer) |

---

### 詳細結果 (Detailed Results)

#### 1. Teleなし (No Tele)
*   **設定**: Front Wide + Left + Right (3カメラ)。Teleカメラを単純に入力から削除。
*   **結果**: 偏差 **6.45m**。
*   **考察**: 明確にカーブを認識できているが、4カメラ時（~9.5m）より精度は落ちた。入力スロット数の変化が影響した可能性がある。
![No Tele Result](../../images/ablation_cam012_f789b390.png)

#### 2. Teleなし・黒埋め (No Tele + Padding)
*   **設定**: Front Wide + Left + Right (3カメラ)。Teleカメラ部分を**黒画像**で埋めて4眼スロットを維持。
*   **結果**: 偏差 **9.50m**。
*   **考察**: **最も良い結果**。ベースラインと同等の性能。Teleカメラの情報（画素）自体は不要だが、4眼の入力構造（またはIndex順序）を維持することが極めて重要であることがわかった。
![No Tele Padding Result](../../images/ablation_cam012_pad_f789b390.png)

#### 3. フロントのみ (Front Only / No Side)
*   **設定**: Front Wide + Front Tele (2カメラ)。サイドカメラ（Left/Right）を削除。
*   **結果**: 偏差 **0.14m**。
*   **考察**: **完全な失敗（直進バイアス）**。サイド情報がないため、自己位置推定やカーブのきつさが認識できない。また、FrontカメラがIndex 0（本来はLeft）に入力されるため、モデルがFront画像をLeft画像と誤認した可能性が高い。
![Front Only Result](../../images/ablation_cam13_f789b390.png)

#### 4. フロントのみ・黒埋め (Front Only + Padding)
*   **設定**: Front Wide + Front Tele (2カメラ)。サイドカメラ部分を**黒画像**で埋めて4眼スロットを維持。
*   **結果**: 偏差 **3.36m**。
*   **考察**: Variable Length (0.1m) よりは改善したが、それでも **3.4m** しか曲がれず、カーブ攻略には不十分。
    *   Frontカメラが正しい位置（Index 1）に入力されたため、最低限の認識は機能した。
    *   しかし、サイドカメラ（Left/Right）からの情報が欠落しているため、急カーブを曲がりきるためのTrajectoryは生成できなかった。
![Front Only Padding Result](../../images/ablation_cam13_pad_f789b390.png)

---

## 3. 考察 (Discussion)
1.  **サイドカメラは必須 (Side Cameras are Essential)**
    *   条件3, 4の結果から、サイドカメラがないと急カーブを曲がりきれないことは明白である。黒画像でスロットを維持しても（条件4）成功には程遠い。
    *   Lidarレスの構成において、横方向の視界（Cross View）確保は最優先事項である。

2.  **Teleカメラは不要 (Tele Camera is Optional)**
    *   条件2の結果（9.5m）は、Teleカメラの情報が完全に欠落（黒画像）していても、他の3カメラがあれば完璧に走行できることを証明している。
    *   近距離のパスプランニングにおいて、遠方の望遠情報は必須ではない。

3.  **入力形式とロバスト性 (Input Format & Robustness)**
    *   **Padding > Variable**: 全てのケースで、単にカメラを減らす（Variable）より、黒画像で埋める（Padding）方が成績が良かった。
    *   モデルは「4つの入力スロット」と「カメラの順序（Index）」に強く依存して学習されている。
    *   実運用上の示唆：カメラ故障時は、可変長入力に切り替えるのではなく、欠損部分を埋めてダミーデータを流す方が安全側（Fail-Soft）に倒れる可能性が高い。

## 4. 実行コマンド (Execution Logs)
再現のための実行コマンドです。

```bash
# 1. Teleなし (Variable)
python test_camera_ablation.py f789b390-1698-4f99-b237-6de4cbbb7666 --cameras 0,1,2

# 2. Teleなし・黒埋め (Padding)
python test_camera_ablation.py f789b390-1698-4f99-b237-6de4cbbb7666 --cameras 0,1,2 --padding

# 3. フロントのみ (Variable)
python test_camera_ablation.py f789b390-1698-4f99-b237-6de4cbbb7666 --cameras 1,3

# 4. フロントのみ・黒埋め (Padding)
python test_camera_ablation.py f789b390-1698-4f99-b237-6de4cbbb7666 --cameras 1,3 --padding
```
