---
title: wandb restore
menu:
  reference:
    identifier: ja-ref-cli-wandb-restore
---

**使い方**

`wandb restore [OPTIONS] RUN`

**概要**

run のコード、config、docker 状態を復元します

**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| `--no-git` | git 状態を復元しない |
| `--branch / --no-branch` | ブランチを作成するか、デタッチされた状態にチェックアウトするか |
| `-p, --project` | アップロードしたいプロジェクト。 |
| `-e, --entity` | リストを特定のエンティティに絞り込むためのエンティティ。 |