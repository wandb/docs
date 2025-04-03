---
title: wandb agent
menu:
  reference:
    identifier: ja-ref-cli-wandb-agent
---

**使用方法**

`wandb agent [OPTIONS] SWEEP_ID`

**概要**

W&B エージェントを実行します。

**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| `-p, --project` | sweep から作成された W&B の run を送信する プロジェクト の名前。 プロジェクト が指定されていない場合、run は「Uncategorized (未分類)」というラベルの プロジェクト に送信されます。 |
| `-e, --entity` | sweep によって作成された W&B の run を送信する ユーザー 名または チーム 名。 指定する entity が既に存在することを確認してください。 entity を指定しない場合、run はデフォルトの entity (通常は ユーザー 名) に送信されます。 |
| `--count` | この エージェント の run の最大数。 |
