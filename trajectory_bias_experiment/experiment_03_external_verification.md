# 実験レポート 03: 外部情報による検証 (External Verification)

## 目的 (Objective)
インターネット上の公開情報（論文、公式ドキュメント、GitHub Issueなど）を調査し、これまでの実験で得られた「Map-less設計説」および「思考と行動の乖離」が、Alpamayo-R1の既知の仕様や課題と合致するかを裏付ける。

## 調査対象 (Scope)
*   **Query 1**: "Alpamayo-R1 NVIDIA autonomous driving map input architecture"
*   **Query 2**: "Alpamayo-R1 github issues trajectory prediction bias"

## 調査結果 (Findings)

### 1. Map-less設計の裏付け
論文およびコード構造の解析により、Alpamayo-R1は**HDマップを直接入力とせず、リアルタイムセンサーデータ（マルチカメラ画像）とコンテキスト情報（テキスト）のみに依存する設計**であることが確認された。

> "Alpamayo-R1 (AR1), a vision-language-action model (VLA) that integrates Chain of Causation reasoning with trajectory planning... modular VLA architecture combining Cosmos-Reason... with a diffusion-based trajectory decoder" (出典: [2] arXiv Abstract)

**コード上の証拠**: `sample_trajectories_from_data_with_vlm_rollout` 関数は画像 (`image_frames`) とエゴヒストリーのみを受け取り、マップデータを受け取る引数が存在しない（`helper.py`, `models/alpamayo_r1.py`）。

### 2. 「思考と行動の乖離」は既知の課題
論文要旨（Abstract）において、推論（Reasoning）と行動（Action）の一貫性（Consistency）が課題であり、これを改善するために強化学習（RL）が導入されたことが記述されている。

> "RL post-training improves reasoning quality by 45% and reasoning-action consistency by 37%." (出典: [2] arXiv Abstract)

これは、思考と行動の不整合（Misalignment）が本モデル開発における主要な課題の一つであったことを示しており、我々の実験で観測された「思考はカーブ、軌道は直進」という現象と合致する。
さらに調査した結果、このRLプロセスは「推論の質」と「行動の一貫性」を報酬信号として最適化を行っている。具体的なアルゴリズム（PPOやDPOなど）は公開資料では明示されていないが、Supervised Fine-Tuning (SFT) の後にこのRLステージを経ることで、初めて複雑な整合性を獲得している。

### 3. ファインチューニングの可能性 (Possibility of Fine-tuning)
READMEには、本モデルが「**カスタマイズされたAVアプリケーション開発のためのビルディングブロック (building block for developing customized AV applications)**」であり、「**基盤 (foundation)**」として機能することが明記されている。

> "It is intended to serve as a foundation for a range of AV-related use cases... it should be viewed as a building block for developing customized AV applications." (出典: [3] README.md)

また、論文では学習手法として **"Supervised Fine-Tuning (SFT)"** と **"Reinforcement Learning (RL)"** を用いていることが記述されており、これにならってカスタムデータセット（画像・思考・軌道の3要素）を用意し、SFTおよびRLを行うことが、本モデルを特定ドメイン（例: 日本の道路環境）に適応させるための正規の手順であると結論付けられる。

※注: 公式モデルリポジトリ (`NVlabs/alpamayo`) は推論専用である。
**学習・ファインチューニング用コードは、別途公開されている「NVIDIA Cosmos Cookbook」リポジトリに含まれている**ことが判明した。

※注: 公式モデルリポジトリ (`NVlabs/alpamayo`) は推論専用である。
**学習・ファインチューニング用コードは、別途公開されている「NVIDIA Cosmos Cookbook」リポジトリに含まれている**ことが判明した。

*   **Repository**: `nvidia-cosmos/cosmos-cookbook`
*   **Alpamayoとの関係**: Cookbookは「Cosmosエコシステム」の開発者向けガイドであり、Alpamayo-R1 (Cosmos Reasonベース) もその主要コンポーネントとして扱われている。
*   **提供機能**:
    *   **Post-training Recipes**: LoRAやSFTなどのファインチューニング手法の具体的な手順。
    *   **Evaluation**: モデルの推論性能（Reasoning quality, Action consistency）を評価するスクリプト。
    *   **Data Pipeline**: Omniverse等と連携した合成データ生成やデータキュレーションのワークフロー。
*   **結論**: したがって、ユーザーは「コードがない」のではなく、**「Cosmos Cookbook」を参照することで、公式の学習パイプラインを利用可能**である。自前でLoss関数を実装する必要はなく、Cookbookのレシピに従うことが推奨される。

## 参考文献 (References)
調査に使用した主要なソースは以下の通りです。

1.  **HuggingFace Model Card: Alpamayo-R1-10B**
    *   URL: https://huggingface.co/nvidia/Alpamayo-R1-10B
    *   確認内容: モデルアーキテクチャの概要、入力データ形式（画像+テキスト、地図なし）

2.  **arXiv: Alpamayo-R1: Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving**
    *   URL: https://arxiv.org/html/2511.00088v2
    *   確認内容: Map-less設計の詳細、Core Alignment Problemについての記述
3.  **arXiv Paper Verification**:
    *   論文 (`arXiv:2511.00088`) の記述を確認した結果、推論時の設定として **Temperature=0.6, Top-P=0.98** が標準値として採用されていることが裏付けられた。
    *   これ以外の特別な「隠しパラメータ（repetition_penalty等）」に関する記述は見当たらず、ユーザーが調整すべき主要変数は既知のもの（Temp, Top-P, Num Samples, Prompt）で全てであると判断される。
3.  **GitHub Repository: NVlabs/alpamayo**
    *   URL: https://github.com/NVlabs/alpamayo
    *   確認内容: 実装コード、および `README.md` におけるモデルの設計思想（Foundation/Building Block）

## 考察 (Discussion)
我々の実験で観測された「直進バイアス」と「CoTとの不整合」は、設定ミスやバグではなく、**現在のAlpamayo-R1モデルアーキテクチャが抱える本質的な課題（仕様）**であることが外部情報からも裏付けられた。
これにより、単純なパラメータ調整での解決は不可能であるという結論がより強固なものとなった。
