# テックブログ構成テンプレート (実験レポート形式) v2

新しいLLMモデルの実験や、技術開発の成果をまとめる際に使用する標準フォーマットです。
**「自分の問題認識を共有し、自分が有効だと思った解決策を提示し、その検証結果（成功・失敗問わず）を論じる」** ことを目的とします。
論文のように完璧な成果である必要はありません。「うまくいかなかったパターン」も貴重な知見として積極的に公開し、読者の技術選定の参考になることを目指します。

---

## 0. TL;DR（3行まとめ）
忙しい読者のために、この記事のハイライトを最初に書きます。
- **What**: 何をしたか（例：Python自作シミュレータを作った）
- **Result**: 結果どうなったか（例：Unity比で100倍高速化した）
- **So What**: それで何が嬉しいのか（例：MLOpsサイクルが爆速になり、開発効率が劇的に向上した）

---

## 1. 目的（問題意識・課題感）
- **What**: 何を解決したいのか、何を知りたいのか。
- **Why**: なぜそれが重要なのか。既存の手法やツールでは何がダメなのか。
- **Goal**: この実験（記事）のゴールは何か。

## 2. アプローチ (Approach)
課題解決のためにどのようなアプローチを取るか、その思考プロセス（Investigation Strategy）を記述します。
単に「これをやりました」ではなく、「なぜそれをやるのか」の戦略を重視します。

**記述のポイント**:
1.  **仮説の列挙**: 問題の原因として考えられる可能性（Alternatives）を挙げる。
2.  **絞り込み**: なぜ他の選択肢ではなく、今回のアプローチを選んだのか（Selection Reason）。
3.  **理論的背景**: そのアプローチを支える理論（Theory）や仮説。

（例：原因として A, B, C が考えられるが、AとBは状況からして考えにくいため、C（今回の手法）を採用する。）

## 3. 前提環境 (Prerequisites)
実験を再現するために必要な環境情報。
- **Hardware**: CPU, GPU, Memory etc.
- **Software**: OS, Python Version, Major Libraries (PyTorch, Autoware version etc.)

## 4. 具体的な検証手順 (Concrete Steps)
アプローチで定めた戦略を実行するための具体的な作業手順やコマンドを記述します。
実行コマンドの直下には、必ずその**実行結果（ログの一部やアウトプット画像）**を貼り付け、その場での確認結果を記述してください。

```bash
# コマンド例
python train.py --config config.yaml
```

**実行結果**:
```text
Epoch 10: Loss = 0.01 (Logの抜粋)
```

これをステップごとに繰り返します。

## 5. Results Summary
**Purpose**: Provide a high-level overview of all findings.
> [!TIP]
> **Micro vs Macro**: While Section 4 handles the "Micro" view (what happened at each step), this section provides the "Macro" view (what it all means together). This separation improves readability by allowing readers to verify details in Section 4 and grasp the big picture here.

| Experiment Case | Outcome | Key Observation |
| :--- | :--- | :--- |
| Case 1 | Failed | Car crashed into wall |
| Case 2 | Success | Smooth navigation |
- 定性的な結果（動画、スクリーンショット、挙動の観察）。
- 成功した点だけでなく、失敗した点もあれば記載。

## 6. ログ詳細（詳細なログが記録されたファイルパス）
- 生データや詳細ログへのリンク。
- MCAPファイルやTensorBoard、WandBのリンクなど、エビデンスとなるもの。

## 7. 考察
- 結果から何が言えるか。
- 方法（作ったもの）は目的（課題解決）に対して有効だったか。
- 残された課題や、次に行うべきアクションは何か。

## 8. 参考文献 (References)
- 関連する論文、公式ドキュメント、GitHubリポジトリへのリンク。
- 記事の信頼性を高めるために記述します。
