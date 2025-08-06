---
title: wandb restore
---

**使い方**

`wandb restore [OPTIONS] RUN`

**概要**

run のコード、config、docker の状態を復元します

**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| `--no-git` | git の状態を復元しません |
| `--branch / --no-branch` | ブランチを作成するか、デタッチされた状態でチェックアウトするか |
| `-p, --project` | アップロード先の Project を指定します。 |
| `-e, --entity` | リストの範囲を Entity で絞り込みます。 |