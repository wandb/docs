---
title: wandb artifact put
menu:
  reference:
    identifier: ja-ref-cli-wandb-artifact-wandb-artifact-put
---

**使い方**

`wandb artifact put [OPTIONS] PATH`

**概要**

アーティファクトを wandb にアップロードします


**オプション**

| **Option** | **Description** |
| :--- | :--- |
| `-n, --name` | プッシュするアーティファクトの名前: project/artifact_name |
| `-d, --description` | このアーティファクトの説明 |
| `-t, --type` | このアーティファクトの種類 |
| `-a, --alias` | このアーティファクトに適用するエイリアス |
| `--id` | アップロード先の run。 |
| `--resume` | 現在の ディレクトリー から最後の run を再開します。 |
| `--skip_cache` | アーティファクト ファイルのアップロード中にキャッシュをスキップします。 |
| `--policy [mutable|immutable]` | アーティファクト ファイルのアップロード中のストレージ ポリシーを設定します。 |