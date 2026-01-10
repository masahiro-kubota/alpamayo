# 検証レポート: 直進バイアスの正体とカメラ構成の重要性 (Camera Configuration Experiment)

## 0. TL;DR
- **What**: **自動運転AIチャレンジのシミュレータ環境（Rosbag）** をAlpamayo-R1に入力したところ、カーブでも直進し続ける「直進バイアス」が発生した。この原因を特定するため、公式データセットを用いた検証とカメラ構成のアブレーション実験を行った。
- **Result**: モデル自体は、正しいカメラ入力さえあれば**半径3.6mの激しいカーブでも曲がれる**能力を持つことが判明。バイアスの主因は、サイドカメラ情報の欠落（または変換ミス）によってモデルが自己位置を見失ったことにあると特定された。
- **So What**: Alpamayo-R1の推論においては、**「4カメラ全視点」かつ「正しい順序」** を維持することが、学習済みのアクション（操舵）を引き出すための絶対条件である。

---

## 1. 目的（問題意識・課題感）
- **What**: 自動運転AIチャレンジのシミュレータ走行データ（Rosbag）をPhysicalAI-AV形式に変換し、Alpamayo-R1に入力したところ、モデルがカーブを認識できず**頑なに直進を選択し続ける現象**が発生した。
- **Why**: 
    - CoT（思考）の一部はカーブに言及していたため、完全に状況が見えていないわけではなさそうだった。
    - 疑わしい要因として、「Rosbagのデータ変換ミス（座標系など）」「モデル自体の学習不足」「入力カメラ構成の違い」の3点が考えられた。
- **Goal**: 変換データの不備を疑う前に、まず公式モデルが**「理想的な条件下なら本当にカーブを曲がれるのか？」**を検証し、その発動条件（必要なカメラ構成）を明らかにする。

## 2. Approach

自動運転AIチャレンジのデータにおいて「直進バイアス」が発生する原因として、以下の3つの可能性が考えられました。

1.  **Parameter Tuning**: Plannerのサンプリング数やホライズン長などのパラメータ調整不足。
2.  **Prompt Engineering**: 「右に曲がれ」といった言語指示（CoT）の誘導力が不足している。
3.  **Input Configuration**: 入力画像の構成（カメラ台数・順序）に不整合がある。

Plannerパラメータやプロンプトは公式サンプル設定から大きく変更していないため、これらが主因である可能性は低いと考えられます。一方で、RosbagからPhysicalAI形式へのデータ変換プロセスにおいては、**カメラの台数不足（サイドカメラが無い等）や入力順序の取り違え**といった構成ミスが最も発生しやすく、かつこれがモデルの空間認識（Spatial Awareness）に致命的な影響を与える可能性があります。

そこで今回は、**「カメラ構成（特にサイドカメラの有無と入力順序）」が推論に与える影響**を重点的に検証するアプローチを採用しました。

なお、検証にあたっては「モデルがそもそもカーブを曲がれるのか」という基礎能力を確認する必要があるため、まずはデータセット全体をスキャンして**「真の急カーブ（半径20m以下）」**を特定し、それをベンチマークとして使用することとしました。

## Theory: カメラ数とステップ数の柔軟性の違い (Spatio-Temporal Flexibility)

Alpamayo-R1において、**「カメラの台数」は自由に変えられるのに、なぜ「過去フレーム数（Step数）」は変えられないのか？** このアーキテクチャの本質的な違いを理解する必要があります。

#### 1. なぜカメラ台数は自由に変更できるのか (Spatial Flexibility)
モデルは画像を **「可変長のトークン列」** として処理するためです。
- **Mechanism**: Vision Encoder (ViT) が各画像をパッチに分割し、トークン化します。
- **Variable Length**:
    - **4カメラ**: `[Left Tokens] + [Front Tokens] + [Right Tokens] + [Tele Tokens]` = 4K トークン
    - **1カメラ**: `[Front Tokens]` = 1K トークン
- **Transformer**: Self-Attention機構は入力長に依存しないため、トークン数が減っても計算可能です。位置情報（Position Encoding）さえ付与されれば、モデルは「空間上の断片的な情報」として処理できます。

#### 2. なぜステップ数は変更できないのか (Temporal Rigidity)
一方で、**「過去何フレーム分を入力するか（History Step）」** は自由に変更できません（例: 4ステップで学習したモデルに1ステップだけ入れることは不可）。
- **Physics Inference (物理量の推定)**: モデルは連続するフレーム間の差分から **「速度」や「加速度」** を推定しています。
    - $v \approx (x_t - x_{t-1}) / \Delta t$
- **Learned Dynamics**: モデルは「4フレーム分の時間発展」という特定のパターンを通じて車両ダイナミクスを学習しています。もし1フレーム（静止画）しか入力しないと、速度情報がゼロになり、モデルは「停止している」と誤認するか、物理法則を無視した挙動（幻覚）を出力します。
- **結論**: 空間情報（カメラ数）は「視野の広さ」の問題なので可変ですが、時間情報（ステップ数）は「物理法則の認識」に直結するため、学習時と厳密に一致させる必要があります。

#### 3. Implicit Positional Bias (暗黙の位置バイアス)
カメラ台数は可変ですが、モデルは学習を通じて **`[Left, Front, Right, Tele]`** という「順序」を強く記憶しています。
- **Risk**: フロントカメラ1枚だけを入力すると、可変長処理により「1番目のトークンブロック」になります。これをモデルは「（1番目だから）左カメラだ」と誤認し、空間認識が歪む原因となります。

## 3. 前提環境 (Prerequisites)
- **Dataset**: [nvidia/PhysicalAI-Autonomous-Vehicles](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
- **Base Model**: Alpamayo-R1 (10B Parameters)

## 4. 具体的な検証手順 (Concrete Steps)

以下の手順で、まずカーブシーンを特定し、その後に実験を行いました。

### Step 1: カーブシーンの探索 (Finding Valid Curves)
公式データセットから「本物のカーブ」を見つけ出すためのスクリプトを実行します。

```bash
# データセット全体をスキャンし、曲率が高い順にリストアップ
python ../../../scan_all_curves.py --threshold 0.05 --max_clips 500 --output ../../logs/curve_scan.json
```

**実行結果**:
- データセット全体（227,985クリップ）のうち、高曲率クリップ (Curvature > 0.05 / 半径 < 20m) はわずか **50個** (全体の **13.2%**) でした。
- この中から、**曲率 0.277 (半径 3.6m)** という最も過酷なクリップ `f789b390` を発見しました。以降の実験ではこれを使用します。
- **Log**: `../../logs/curve_scan_500samples.json`

```bash
# 特定されたクリップの曲率詳細を確認
python find_curve_clips.py --top_n 10
```

### Step 2: カメラ構成のアブレーション実験
特定された急カーブ (`f789b390`) に対し、以下の4つの条件で推論実験を行いました。

#### Case 1: Teleなし (Standard Variable)
Teleカメラを単純に入力から削除し、3眼（Left, Front, Right）で推論します。

```bash
# 1. Teleなし (Variable)
python test_camera_ablation.py f789b390-1698-4f99-b237-6de4cbbb7666 --cameras 0,1,2
```

**実行結果**:
- **Max Deviation**: **6.45 m**
- **判定**: **Sueccess**。カーブを認識して曲がれているが、4眼フル（~9.5m）よりは精度低下。
- **Log**: [`../../logs/case1_no_tele.log`](../../logs/case1_no_tele.log)
![No Tele Result](../../images/ablation_cam012_f789b390.png)

#### Case 2: Teleなし・黒埋め (Fail-Soft Padding)
Teleカメラを削除するが、入力配列の長さを変えず、黒画像で埋めます。

```bash
# 2. Teleなし・黒埋め (Padding)
python test_camera_ablation.py f789b390-1698-4f99-b237-6de4cbbb7666 --cameras 0,1,2 --padding
```

**実行結果**:
- **Max Deviation**: **9.50 m**
- **判定**: **Perfect**。ベースラインと同等の最高性能。Teleの画素情報は不要だが、4眼の入力構造維持が重要であることを示唆。
- **Log**: [`../../logs/case2_no_tele_pad.log`](../../logs/case2_no_tele_pad.log)
![No Tele Padding Result](../../images/ablation_cam012_pad_f789b390.png)

#### Case 3: フロントのみ (Front Only / Variable)
サイドカメラ（Left/Right）とTeleカメラを削除し、Front Wide（120°）**1枚のみ**を入力します。

```bash
# 3. フロントのみ (Variable)
# Index 1: Front Wide
python test_camera_ablation.py f789b390-1698-4f99-b237-6de4cbbb7666 --cameras 1
```

**実行結果**:
- **Max Deviation**: **0.01 m**
- **判定**: **Failure** (完全直進)。
- 思考（Reasoning）では「右カーブ」と言っているが、行動（Action）は直進。インデックスずれ（Front画像がIndex 0に入り、Left画像として誤認された）が疑われる。
- **Log**: [`../../logs/case3_front_only.log`](../../logs/case3_front_only.log)
![Front Only Result](../../images/ablation_cam13_f789b390.png)

#### Case 4: フロントのみ・黒埋め (Front Only / Padding)
Front Wideのみを残し、他3枚（Left, Right, Tele）を黒画像で埋めます。

```bash
# 4. フロントのみ・黒埋め (Padding)
python test_camera_ablation.py f789b390-1698-4f99-b237-6de4cbbb7666 --cameras 1 --padding
```

**実行結果**:
- **Max Deviation**: **0.42 m**
- **判定**: **Partial**。Variable (0.14m) よりはマシだが、曲がりきれず。サイドカメラの幾何学的情報が必須であることを証明。
- **Log**: [`../../logs/case4_front_only_pad.log`](../../logs/case4_front_only_pad.log)
![Front Only Padding Result](../../images/ablation_cam13_pad_f789b390.png)

#### Case 5: 順序入れ替え (Permuted Order)
入力画像の順序を意図的に入れ替えることで、モデルが「どのスロットにどのカメラが入っているか」をIndex順序に依存して判断しているかを検証します。

```bash
# 5. 順序入れ替え (Front, Left, Right, Tele)
# Regular: Left(0), Front(1), Right(2), Tele(3) -> Permuted: Front(1), Left(0), Right(2), Tele(3)
python test_camera_ablation.py f789b390-1698-4f99-b237-6de4cbbb7666 --cameras 1,0,2,3
```

**実行結果**:
- **Log**: [`../../logs/case5_permuted.log`](../../logs/case5_permuted.log)
- **Max Deviation**: **9.23 m**
- **判定**: **Success** (Surprisingly)。
- **考察**: 画像の入力順序を入れ替えても、パディングなしのVariable Length入力においてモデルはある程度のロバスト性を示しました。これは、モデルが位置埋め込み（Positional Embedding）や視覚特徴から各カメラの役割をある程度識別できている可能性、あるいはこの特定の並び替え（Front <-> Left）が致命的ではなかった可能性を示唆しています。しかし、標準順序（Case 2: 9.50m）よりはわずかに性能が低下しています。
![Permuted Order Result](../../images/ablation_cam1023_f789b390.png)

## 5. 結果のまとめ (Results Summary)

4つのアブレーション実験の結果をまとめると以下の通りです。
サイドカメラ（Left/Right）の欠損が最も致命的であり（Case 3, 4）、カメラを減らす場合でも黒画像パディング（Padding）で構造を維持することがロバスト性に寄与することがわかります。
また、Case 5の順序入れ替え実験では、意外にも高い性能（9.23m）が維持されましたが、やはり標準順序（9.50m）が最も高いパフォーマンスを発揮しました。

| ID | 条件 (Condition) | 入力形式 (Input Type) | カメラ構成 (Cameras) | Max Dev | 結果 (Result) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **Teleなし** | Variable Length | Left, Front, Right | **6.45 m** | **Success** (Curved) |
| **2** | **Teleなし・黒埋め** | Padding (Black) | Left, Front, Right, *Black* | **9.50 m** | **Success** (Perfect) |
| **3** | **フロントのみ** | Variable Length | Front, Tele | **0.01 m** | **Failure** (Straight) |
| **4** | **フロントのみ・黒埋め** | Padding (Black) | *Black*, Front, *Black*, Tele | **0.42 m** | **Failure** (Straight) |
| **5** | **順序入れ替え** | Permuted | Front, Left, Right, Tele | **9.23 m** | **Success** (Good) |

| ID | 条件 (Condition) | 入力形式 | カメラ構成 | Max Deviation | 結果 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **Teleなし** | Variable | Left, Front, Right | **6.45 m** | **Success** (曲がれた) |
| **2** | **Teleなし・黒埋め** | Padding | Left, Front, Right, *Black* | **9.50 m** | **Perfect** (完走) |
| **3** | **フロントのみ** | Variable | Front, Tele | **0.14 m** | **Failure** (完全直進) |
| **4** | **フロントのみ・黒埋め** | Padding | *Black*, Front, *Black*, Tele | **3.36 m** | **Partial** (曲がりきれず) |

## 7. 考察

### Why Front-Only Failed so Badly? (0.14m vs 9.50m)
フロントカメラのみの場合、偏差が **0.14m** という衝撃的な低さ（完全直進）になりました。
これには2つの要因が複合しています。

1.  **幾何学的情報の欠落**:
    - サイドカメラからの視差情報がないため、モデルは自車が「車線内のどこにいるか」という横方向の位置（Lateral Position）を正確に推定できません。不確実性が高い時、拡散モデル（Diffusion Policy）は最も安全で保守的な「平均値（＝直進）」を出力する傾向があります。

2.  **インデックスずれ (Index Mismatch)**:
    - Variable Length入力（Case 3）では、フロント画像を配列の先頭（Index 0）に入れてしまいます。
    - モデルは `Index 0 = Left Camera` という強い事前分布を持っているため、フロント画像を「左側面を見ている画像」として処理してしまった可能性があります。これにより空間認識が90度回転し、制御不能に陥りました。

### Conclusion
- **自動運転AIチャレンジの攻略法**: シミュレータデータを入力にする際、単純に「フロント画像だけ」を入れないこと。必ず4視点を用意するか、少なくとも **黒画像パディング (Case 4)** で位置インデックスを合わせる必要がある。
- **入力順序は厳守**: たとえカメラが減っても、順序を詰めてはいけません（Variableは危険）。黒画像パディング（Padding）の方が、Attentionの無駄遣い（Softmax問題）という副作用はあるものの、**位置情報の保持**というメリットが勝り、はるかにマシな結果（3.36m vs 0.14m）を出しました。

### Next Steps
- 推論パイプラインにおいて、カメラ欠損時は動的リサイズではなく「黒画像パディング」または「直前のフレーム保持」を行うフェイルセーフを実装する。
- Prompt Engineeringではなく、**Input Engineering**（入力テンソルの設計）こそがVLAモデルの制御には重要である。
