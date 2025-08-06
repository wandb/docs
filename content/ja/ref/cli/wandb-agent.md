---
title: wandb エージェント
---

**使用方法**

`wandb agent [OPTIONS] SWEEP_ID`

**概要**

W&B エージェントを実行します


**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| `-p, --project` | sweep から作成される W&B Run を送信する Project の名前です。Project を指定しない場合、Run は 'Uncategorized' というラベルの Project に送信されます。 |
| `-e, --entity` | sweep で作成された W&B Run を送信したい Entity（ユーザー名またはチーム名）を指定します。指定した Entity が既に存在していることを確認してください。Entity を指定しない場合、通常はあなたのユーザー名がデフォルト Entity となり、そこに Run が送信されます。 |
| `--count` | このエージェントが実行する Run の最大数です。 |