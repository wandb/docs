---
title: wandb beta sync
menu:
  reference:
    identifier: ja-ref-cli-wandb-beta-wandb-beta-sync
---

**使用方法**

`wandb beta sync [OPTIONS] WANDB_DIR`

**概要**

トレーニング run を W&B にアップロードします。

**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| `--id` | アップロード先の run。 |
| `-p, --project` | アップロード先の project。 |
| `-e, --entity` | スコープする entity。 |
| `--skip-console` | コンソール ログをスキップします。 |
| `--append` | run を追加します。 |
| `-i, --include` | 含める glob。複数回使用できます。 |
| `-e, --exclude` | 除外する glob。複数回使用できます。 |
| `--mark-synced / --no-mark-synced` | run を同期済みとしてマークします。 |
| `--skip-synced / --no-skip-synced` | 同期済みの run をスキップします。 |
| `--dry-run` | 何もアップロードせずに dry run を実行します。 |
