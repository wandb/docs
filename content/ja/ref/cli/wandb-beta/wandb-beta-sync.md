---
title: wandb beta sync
menu:
  reference:
    identifier: ja-ref-cli-wandb-beta-wandb-beta-sync
---

**使い方**

`wandb beta sync [OPTIONS] WANDB_DIR`

**概要**

トレーニング run を W&B へアップロードします

**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| `--id` | アップロードしたい run の ID です。 |
| `-p, --project` | アップロードしたい Project です。 |
| `-e, --entity` | スコープする Entity です。 |
| `--skip-console` | コンソールログをスキップします |
| `--append` | run を追加します |
| `-i, --include` | 含めるファイルのグロブ指定。複数回使用可能です。 |
| `-e, --exclude` | 除外するファイルのグロブ指定。複数回使用可能です。 |
| `--mark-synced / --no-mark-synced` | run を同期済みとしてマークします |
| `--skip-synced / --no-skip-synced` | 同期済みの run をスキップします |
| `--dry-run` | 何もアップロードせずにドライランを実行します。 |