---
title: wandb artifact put
menu:
  reference:
    identifier: ja-ref-cli-wandb-artifact-wandb-artifact-put
---

**使用法**

`wandb artifact put [OPTIONS] PATH`

**概要**

アーティファクト を wandb にアップロードします。

**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| `-n, --name` | プッシュする アーティファクト の名前: project/artifact_name |
| `-d, --description` | この アーティファクト の説明 |
| `-t, --type` | アーティファクト の種類 |
| `-a, --alias` | この アーティファクト に適用する エイリアス |
| `--id` | アップロード先の run 。 |
| `--resume` | 現在の ディレクトリー から最後の run を再開します。 |
| `--skip_cache` | アーティファクト ファイルのアップロード中にキャッシュをスキップします。 |
| `--policy [mutable\|immutable]` | アーティファクト ファイルのアップロード中にストレージポリシーを設定します。 |
