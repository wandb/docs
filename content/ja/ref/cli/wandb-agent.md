---
title: wandb agent
menu:
  reference:
    identifier: ja-ref-cli-wandb-agent
---

**使い方**

`wandb agent [OPTIONS] SWEEP_ID`

**概要**

W&B エージェントを実行します。


**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| `-p, --project` | sweep で作成された W&B の run の送信先にする Project 名です。Project を指定しない場合、run は 'Uncategorized' というラベルの Project に送信されます。 |
| `-e, --entity` | sweep によって作成された W&B の run の送信先にする、ユーザー名または Team 名です。指定した Entity が既に存在していることを確認してください。Entity を指定しない場合、run はあなたのデフォルトの Entity（通常はあなたのユーザー名）に送信されます。 |
| `--count` | この エージェント が実行する run の最大数です。 |