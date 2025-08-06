---
title: wandb artifact put
---

**使用方法**

`wandb artifact put [OPTIONS] PATH`

**概要**

wandb にアーティファクトをアップロードします

**オプション**

| **Option** | **Description** |
| :--- | :--- |
| `-n, --name` | プッシュするアーティファクトの名前：  project/artifact_name |
| `-d, --description` | このアーティファクトの説明 |
| `-t, --type` | アーティファクトのタイプ |
| `-a, --alias` | このアーティファクトに適用するエイリアス |
| `--id` | アップロードしたい run の指定 |
| `--resume` | 現在のディレクトリーから前回の run を再開します。 |
| `--skip_cache` | アーティファクトファイルのアップロード時にキャッシュをスキップします。 |
| `--policy [mutable|immutable]` | アーティファクトファイルをアップロードする際のストレージポリシーを設定します。 |