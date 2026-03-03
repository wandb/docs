---
title: テンプレート
---

<div id="agent-prompt-task-title">
  # エージェント プロンプト: [タスクタイトル]
</div>

<div id="requirements">
  ## 要件
</div>

このタスクを開始する前に満たしておく必要があるアクセス要件や前提条件を次に示します。

- [ ] 必要なシステムアクセス（例: W&B 社員としてのアクセス権）。
- [ ] 必要な権限（例: リポジトリへの書き込み権限）。
- [ ] 必要なツールまたは依存関係。

<div id="agent-prerequisites">
  ## エージェントの前提条件
</div>

開始する前にユーザーから取得しておく情報:

1. **[Required info 1]** - その情報が必要な理由
2. **[Required info 2]** - その情報が必要な理由
3. **[Optional info]** - その情報がいつ／なぜ必要になる可能性があるか

<div id="task-overview">
  ## タスクの概要
</div>

このランブックで行うことと、どのような状況で使用するかの簡潔な説明。

> **Note**: あらかじめユーザーが知っておくべき重要なコンテキストや制約。

<div id="context-and-constraints">
  ## コンテキストと制約事項
</div>

<div id="systemtool-limitations">
  ### システム／ツールの制限事項
</div>

- 制限事項 1 と、それが本タスクに与える影響
- 制限事項 2 と、回避策（ある場合）

<div id="important-context">
  ### 重要な前提事項
</div>

- 主要な背景情報
- よくあるハマりどころやエッジケース
- セキュリティ上の考慮事項

<div id="step-by-step-process">
  ## 手順
</div>

<div id="1-first-major-step">
  ### 1. [First major step]
</div>

このステップで達成される内容の説明。

```bash
# コマンド例
command --with-flags
```

**想定される結果**: このステップ完了後に想定される結果。


<div id="2-second-major-step">
  ### 2. [2つ目の主要なステップ]
</div>

説明と、必要な意思決定ポイント。

**Agent note**: AI エージェント向けの特別な指示。例:

- ユーザーに確認や追加説明を求めるタイミング
- 権限がない場合に実行するフォールバック手順
- よくあるパターンの扱い方

<div id="3-continue-with-remaining-steps">
  ### 3. [残りの手順に進む...]
</div>

<div id="verification-and-testing">
  ## 検証とテスト
</div>

期待される結果:

- ✓ 成功の目安 1
- ✓ 成功の目安 2
- ✗ 一般的な失敗時の兆候とその意味

<div id="how-to-verify-success">
  ### 成功したかを確認する方法
</div>

1. 次の点をチェックする…
2. 次の点を確認する…
3. 次の方法でテストする…

<div id="common-issues-and-solutions">
  ## よくある問題とその解決方法
</div>

<div id="issue-common-problem-1">
  ### 問題: [Common problem 1]
</div>

- **症状**: この問題がどのような形で現れるか
- **原因**: なぜ発生するのか
- **解決方法**: ステップごとの解決手順

<div id="issue-common-problem-2">
  ### 問題: [よくある問題 2]
</div>

- **症状**: 
- **原因**: 
- **解決方法**: 

<div id="cleanup-instructions">
  ## クリーンアップ手順
</div>

タスクが完了したら、以下を実行してください。

1. 一時ファイルや一時ブランチを削除します。
2. 変更した設定を元に戻します。
3. 行った恒久的な変更を記録します。

```bash
# クリーンアップコマンドの例
git branch -D temp-branch-name
rm -f temporary-files
```


<div id="checklist">
  ## チェックリスト
</div>

プロセス全体の概要チェックリスト：

- [ ] すべての要件を満たした。
- [ ] ユーザーから必要な情報を収集した。
- [ ] ステップ 1 を完了した：[簡単な説明]。
- [ ] ステップ 2 を完了した：[簡単な説明]。
- [ ] 結果を検証した。
- [ ] 一時的なリソースをクリーンアップした。
- [ ] 恒久的な変更をすべて記録した。

<div id="notes">
  ## 補足
</div>

- 追加のヒントや補足情報。
- 関連ドキュメントへのリンク。
- 代替手法を利用するタイミング。