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
| `-p, --project` | sweep から作成された W&B runs が送信されるプロジェクトの名前。プロジェクトが指定されていない場合、run は「Uncategorized」というラベルのプロジェクトに送信されます。 |
| `-e, --entity` | sweep によって作成された W&B runs を送信したいユーザー名またはチーム名。指定した entity が既に存在することを確認してください。entity を指定しない場合、run は通常ユーザー名であるデフォルトの entity に送信されます。 |
| `--count` | このエージェントの最大 run 数。 |