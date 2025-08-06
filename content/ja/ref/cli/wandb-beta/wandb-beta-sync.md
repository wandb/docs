---
title: wandb beta sync
---

**使い方**

`wandb beta sync [OPTIONS] WANDB_DIR`

**概要**

トレーニング run を W&B にアップロードします

**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| `--id` | アップロードしたい run の ID。 |
| `-p, --project` | アップロード先の Project。 |
| `-e, --entity` | 対象とする Entity。 |
| `--skip-console` | コンソールログをスキップします |
| `--append` | run を追加します |
| `-i, --include` | 含めるファイルのグロブパターン。複数回指定可能。 |
| `-e, --exclude` | 除外するファイルのグロブパターン。複数回指定可能。 |
| `--mark-synced / --no-mark-synced` | run を同期済みとしてマークします |
| `--skip-synced / --no-skip-synced` | 同期済みの run をスキップします |
| `--dry-run` | アップロードせずにドライランを実行します。 |