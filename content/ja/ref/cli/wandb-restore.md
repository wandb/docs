---
title: wandb restore
menu:
  reference:
    identifier: ja-ref-cli-wandb-restore
---

**使用法**

`wandb restore [OPTIONS] RUN`

**概要**

run のコード、config、および Docker の状態を復元します。

**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| `--no-git` | git の状態を復元しない |
| `--branch / --no-branch` | ブランチを作成するか、デタッチしてチェックアウトするか |
| `-p, --project` | アップロード先の project 。 |
| `-e, --entity` | リストのスコープとなる entity 。 |
