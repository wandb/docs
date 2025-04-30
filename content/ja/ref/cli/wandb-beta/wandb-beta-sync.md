---
title: wandb ベータ 同期
menu:
  reference:
    identifier: ja-ref-cli-wandb-beta-wandb-beta-sync
---

**使用法**

`wandb beta sync [OPTIONS] WANDB_DIR`

**概要**

トレーニング run を W&B にアップロードします

**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| `--id` | アップロードしたい run です。 |
| `-p, --project` | アップロードしたい project です。 |
| `-e, --entity` | スコープにする entity です。 |
| `--skip-console` | コンソールログをスキップします |
| `--append` | run を追加します |
| `-i, --include` | 含めるためのグロブ。複数回使用可能です。 |
| `-e, --exclude` | 除外するためのグロブ。複数回使用可能です。 |
| `--mark-synced / --no-mark-synced` | run を同期済みとしてマークします |
| `--skip-synced / --no-skip-synced` | 同期済みの run をスキップします |
| `--dry-run` | 何もアップロードせずにドライ run を実行します。 |