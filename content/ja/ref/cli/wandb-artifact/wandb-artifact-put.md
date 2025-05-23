---
title: wandb アーティファクト put
menu:
  reference:
    identifier: ja-ref-cli-wandb-artifact-wandb-artifact-put
---

**使用方法**

`wandb artifact put [OPTIONS] PATH`

**概要**

アーティファクトを wandb にアップロードします

**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| `-n, --name` | プッシュするアーティファクトの名前:   project/artifact_name |
| `-d, --description` | このアーティファクトの説明 |
| `-t, --type` | アーティファクトのタイプ |
| `-a, --alias` | このアーティファクトに適用するエイリアス |
| `--id` | アップロードしたい run を指定します。 |
| `--resume` | 現在のディレクトリーから前回の run を再開します。 |
| `--skip_cache` | アーティファクトファイルのアップロード中にキャッシュをスキップします。 |
| `--policy [mutable\|immutable]` | アーティファクトファイルをアップロードする際のストレージポリシーを設定します。 |