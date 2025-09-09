---
title: wandb restore
menu:
  reference:
    identifier: ja-ref-cli-wandb-restore
---

**使い方**

`wandb restore [OPTIONS] RUN`

**概要**

run のコード、設定、Docker の状態を復元します。


**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| `--no-git` | Git の状態を復元しません。 |
| `--branch / --no-branch` | ブランチを作成するか、detached でチェックアウトするかを指定します。 |
| `-p, --project` | アップロード先の Project を指定します。 |
| `-e, --entity` | 一覧の対象とする Entity を指定します。 |