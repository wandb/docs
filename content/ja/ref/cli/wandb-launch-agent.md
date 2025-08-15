---
title: wandb launch-agent
menu:
  reference:
    identifier: ja-ref-cli-wandb-launch-agent
---

**使い方**

`wandb launch-agent [OPTIONS]`

**概要**

W&B Launch エージェントを実行します。

**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| `-q, --queue` | エージェントが監視するキュー名を指定します。複数の -q フラグに対応しています。 |
| `-e, --entity` | 使用する Entity を指定します。デフォルトは現在ログイン中のユーザーです。 |
| `-l, --log-file` | エージェント内部ログの出力先を指定します。- を指定すると標準出力となります。デフォルトでは、すべてのエージェントログは wandb/サブディレクトリー内の debug.log か、WANDB_DIR が設定されていればそちらに保存されます。 |
| `-j, --max-jobs` | このエージェントが並列で実行できる Launch ジョブの最大数です。デフォルトは1。-1に設定すると上限がなくなります。|
| `-c, --config` | 使用するエージェント設定 yaml ファイルのパスを指定します。|
| `-v, --verbose` | 詳細な出力を表示します。|