---
title: wandb restore
menu:
  reference:
    identifier: ja-ref-cli-wandb-restore
---

**使い方**

`wandb restore [OPTIONS] RUN`

**概要**

run の コード、config、docker の状態を復元します。

**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| `--no-git` | git の状態を復元しません |
| `--branch / --no-branch` | ブランチを作成するか、detached チェックアウトするか |
| `-p, --project` | アップロード先の Project を指定します。 |
| `-e, --entity` | 一覧表示の範囲を指定する Entity。 |