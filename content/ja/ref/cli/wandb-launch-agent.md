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
| `-q, --queue` | エージェントが監視するキュー名。-q フラグは複数指定できます。 |
| `-e, --entity` | 使用する Entity。デフォルトは現在ログイン中のユーザーです。 |
| `-l, --log-file` | エージェントの内部ログの出力先。標準出力に出力するには - を使用します。デフォルトでは、すべてのエージェントのログは wandb/ サブディレクトリーの debug.log に出力されるか、WANDB_DIR が設定されていればその場所に出力されます。 |
| `-j, --max-jobs` | このエージェントが並列実行できる Launch ジョブの最大数。デフォルトは 1。上限なしにするには -1 を設定してください。 |
| `-c, --config` | 使用するエージェントの config YAML へのパス。 |
| `-v, --verbose` | 詳細な出力を表示します。 |