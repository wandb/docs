---
title: テンプレート
---

<div id="agent-prompt-task-title">
  # エージェント プロンプト: [タスクタイトル]
</div>

<div id="requirements">
  ## 必要条件
</div>

このタスクを開始する前に満たす必要があるアクセス要件または前提条件を列挙してください：

* [ ] 必須のシステムアクセス (例：W&amp;B 従業員アクセス) 。
* [ ] 必須の権限 (例：リポジトリへの書き込みアクセス) 。
* [ ] 必須のツールまたは依存関係。

<div id="agent-prerequisites">
  ## エージェントの前提条件
</div>

開始する前にユーザーから収集しておく情報:

1. **[Required info 1]** - その情報が必要な理由
2. **[Required info 2]** - その情報が必要な理由
3. **[Optional info]** - その情報がいつ・なぜ必要になるか

<div id="task-overview">
  ## タスク概要
</div>

この runbook が何を行うものか、またどのような場面で使用するかを簡潔に記載します。

> **Note**: 文書の冒頭でユーザーに伝えておくべき重要な前提や制約事項を記載します。

<div id="context-and-constraints">
  ## コンテキストと制約条件
</div>

<div id="systemtool-limitations">
  ### システム/ツールの制限事項
</div>

* 制限1とそのタスクへの影響
* 制限2と (あれば) 回避策

<div id="important-context">
  ### 重要な前提事項
</div>

* 重要な背景情報
* よくある落とし穴やエッジケース
* セキュリティ上の考慮事項

<div id="step-by-step-process">
  ## 手順の流れ
</div>

<div id="1-first-major-step">
  ### 1. [最初の大きなstep]
</div>

このstepで達成できることの説明です。

```bash
# コマンド例
command --with-flags
```

**期待される結果**: このstepの実行後に想定される結果。


<div id="2-second-major-step">
  ### 2. [2 番目の主要 step]
</div>

説明および意思決定が必要となるポイントを記載します。

**Agent note**: AI エージェント向けの特別な指示。たとえば:

* ユーザーに説明や確認を求めるタイミング
* 権限が不足している場合のフォールバック手順
* 一般的なバリエーションへの対応方法

<div id="3-continue-with-remaining-steps">
  ### 3. [残りの手順を実行する...]
</div>

<div id="verification-and-testing">
  ## 検証とテスト
</div>

想定される結果：

- ✓ 成功指標 1
- ✓ 成功指標 2
- ✗ 一般的な失敗指標とその意味

<div id="how-to-verify-success">
  ### 成功を確認する方法
</div>

1. …を確認します
2. …であることを確認します
3. …してテストします

<div id="common-issues-and-solutions">
  ## よくある問題とその解決方法
</div>

<div id="issue-common-problem-1">
  ### Issue: [Common problem 1]
</div>

* **Symptoms**: この問題の症状
* **Cause**: 発生原因
* **Solution**: 手順に沿った解決方法

<div id="issue-common-problem-2">
  ### 問題: [Common problem 2]
</div>

- **症状**: 
- **原因**: 
- **解決方法**: 

<div id="cleanup-instructions">
  ## クリーンアップ手順
</div>

タスクを完了したら、以下を実行します。

1. 一時ファイルや一時ブランチをすべて削除します。
2. 変更した設定を元に戻します。
3. 行った恒久的な変更内容を記録します。

```bash
# クリーンアップコマンドの例
git branch -D temp-branch-name
rm -f temporary-files
```


<div id="checklist">
  ## チェックリスト
</div>

プロセス全体の概要チェックリスト：

* [ ] すべての要件を満たしている。
* [ ] ユーザーから必要な情報を収集している。
* [ ] step 1 を完了している：[簡単な説明]。
* [ ] step 2 を完了している：[簡単な説明]。
* [ ] 結果を検証している。
* [ ] 一時的なリソースをクリーンアップしている。
* [ ] 恒久的な変更点を記録している。

<div id="notes">
  ## メモ
</div>

* 追加のヒントや背景情報。
* 関連ドキュメントへのリンク。
* 代替アプローチを使用するタイミング。