---
title: wandb エージェント
menu:
  reference:
    identifier: ja-ref-cli-wandb-agent
---

**使用方法**

`wandb agent [OPTIONS] SWEEP_ID`

**概要**

W&B エージェントを実行します

**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| `-p, --project` | sweep から作成された W&B Runs を送信する Project 名を指定します。Project を指定しない場合、Run は「Uncategorized」という Project に送信されます。 |
| `-e, --entity` | sweep で作成された W&B Runs を送信したい Users または Teams の名前を指定します。指定した Entity が既に存在していることを確認してください。Entity を指定しない場合、デフォルトの Entity（通常はあなたのユーザー名）に送信されます。 |
| `--count` | このエージェントで実行する Run の最大数。 |