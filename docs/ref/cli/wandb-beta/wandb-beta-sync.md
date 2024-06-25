
# wandb beta sync

**使用方法**

`wandb beta sync [OPTIONS] WANDB_DIR`

**概要**

トレーニング run を W&B にアップロードする

**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| --id | アップロードしたい run のID |
| -p, --project | アップロードしたい project |
| -e, --entity | スコープにする entity |
| --skip-console | コンソールログをスキップ |
| --append | run を追加 |
| -i, --include | 含めるグロブパターン。複数回使用可能。 |
| -e, --exclude | 除外するグロブパターン。複数回使用可能。 |
| --mark-synced / --no-mark-synced | run を同期済みとしてマーク |
| --skip-synced / --no-skip-synced | 同期済みの run をスキップ |
| --dry-run | アップロードをせずにドライランを実行 |