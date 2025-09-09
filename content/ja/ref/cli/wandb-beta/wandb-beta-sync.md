---
title: wandb beta sync
menu:
  reference:
    identifier: ja-ref-cli-wandb-beta-wandb-beta-sync
---

**使用方法**

`wandb beta sync [OPTIONS] WANDB_DIR`

**概要**

トレーニング run を W&B にアップロード


**オプション**

| **Option** | **Description** |
| :--- | :--- |
| `--id` | アップロード先の run。 |
| `-p, --project` | アップロード先の プロジェクト。 |
| `-e, --entity` | スコープ対象の Entity。 |
| `--skip-console` | コンソール ログをスキップ |
| `--append` | run に追記 |
| `-i, --include` | 含める glob パターン。複数回使用できます。 |
| `-e, --exclude` | 除外する glob パターン。複数回使用できます。 |
| `--mark-synced / --no-mark-synced` | runs を同期済みとしてマーク |
| `--skip-synced / --no-skip-synced` | 同期済み runs をスキップ |
| `--dry-run` | 何もアップロードせずに dry run を実行。 |