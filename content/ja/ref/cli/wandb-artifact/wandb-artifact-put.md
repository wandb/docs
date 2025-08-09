---
title: wandb artifact put
menu:
  reference:
    identifier: ja-ref-cli-wandb-artifact-wandb-artifact-put
---

**使い方**

`wandb artifact put [OPTIONS] PATH`

**概要**

artifact を wandb にアップロードします。


**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| `-n, --name` | プッシュする artifact の名前：project/artifact_name |
| `-d, --description` | この artifact の説明 |
| `-t, --type` | artifact のタイプ |
| `-a, --alias` | この artifact に適用するエイリアス |
| `--id` | アップロードしたい run を指定します。 |
| `--resume` | 現在のディレクトリーから直前の run を再開します。 |
| `--skip_cache` | artifact ファイルをアップロードする際にキャッシュをスキップします。 |
| `--policy [mutable|immutable]` | artifact ファイルをアップロードする際のストレージポリシーを設定します。 |