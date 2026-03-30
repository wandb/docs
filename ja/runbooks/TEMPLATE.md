---
title: テンプレート
---

<div id="agent-prompt-task-title">
  # エージェントへのプロンプト: [タスクタイトル]
</div>

<div id="requirements">
  ## 要件
</div>

このタスクを開始する前に満たしておく必要があるアクセス要件や前提条件を記載してください。

* [ ] 必須のシステムアクセス (例: W&amp;B従業員向けアクセス) 。
* [ ] 必須の権限 (例: リポジトリへの書き込み権限) 。
* [ ] 必須のツールまたは依存関係。

<div id="agent-prerequisites">
  ## エージェント の事前準備
</div>

開始前にユーザーから収集する情報:

1. **[必須情報 1]** - 必要な理由
2. **[必須情報 2]** - 必要な理由
3. **[任意情報]** - 必要になる場合やその理由

<div id="task-overview">
  ## タスク概要
</div>

このランブックの目的と、どのような場合に使用するかを簡潔に説明します。

> **注**: ユーザーが事前に知っておくべき重要な前提や制限事項。

<div id="context-and-constraints">
  ## 前提と制約
</div>

<div id="systemtool-limitations">
  ### システム/ツールの制約
</div>

* 制約 1 と、それがタスクに与える影響
* 制約 2 と、該当する場合の回避策

<div id="important-context">
  ### 重要な前提情報
</div>

* 重要な背景情報
* よくある落とし穴やエッジケース
* セキュリティ上の考慮事項

<div id="step-by-step-process">
  ## stepごとの手順
</div>

<div id="1-first-major-step">
  ### 1. [最初の主要なstep]
</div>

このstepで達成する内容の説明。

```bash
# コマンド例
command --with-flags
```

**期待される結果**: この step の後に起こるべきこと。


<div id="2-second-major-step">
  ### 2. [2つ目の主要な step]
</div>

説明と、必要に応じた判断ポイント。

**エージェント note**: 次のような AI agent 向けの特別な指示:

* ユーザーに確認を求めるタイミング
* 権限が不足している場合のフォールバック手順
* よくあるバリエーションへの対応方法

<div id="3-continue-with-remaining-steps">
  ### 3. [残りのstepに進みます...]
</div>

<div id="verification-and-testing">
  ## 検証とテスト
</div>

想定される結果:

* ✓ 成功の指標 1
* ✓ 成功の指標 2
* ✗ よくある失敗の指標とその意味

<div id="how-to-verify-success">
  ### 成功したことを確認する方法
</div>

1. 次の点を確認します...
2. 次の点を確認します...
3. 次の方法でテストします...

<div id="common-issues-and-solutions">
  ## よくある問題と解決策
</div>

<div id="issue-common-problem-1">
  ### 問題: [一般的な問題 1]
</div>

* **症状**: この問題の現れ方
* **原因**: 発生する理由
* **解決策**: stepごとの対処方法

<div id="issue-common-problem-2">
  ### 問題: [よくある問題 2]
</div>

* **症状**:
* **原因**:
* **解決方法**: 

<div id="cleanup-instructions">
  ## クリーンアップ手順
</div>

タスク完了後:

1. 一時ファイルやブランチを削除します。
2. 変更した設定を元に戻します。
3. 恒久的な変更があれば記録します。

```bash
# クリーンアップコマンドの例
git branch -D temp-branch-name
rm -f temporary-files
```


<div id="checklist">
  ## チェックリスト
</div>

プロセス全体の確認用チェックリスト:

* [ ] すべての要件を満たした。
* [ ] ユーザーから必要な情報を収集した。
* [ ] step 1 を完了した: [簡単な説明]。
* [ ] step 2 を完了した: [簡単な説明]。
* [ ] 結果を確認した。
* [ ] 一時リソースを削除した。
* [ ] 恒久的な変更があれば文書化した。

<div id="notes">
  ## メモ
</div>

* 追加のヒントや補足情報。
* 関連ドキュメントへのリンク。
* 代替アプローチを使用する場面。