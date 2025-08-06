---
title: wandb sync
---

**使い方**

`wandb sync [OPTIONS] [PATH]...`

**概要**

オフライントレーニングディレクトリーを W&B にアップロードします

**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| `--id` | アップロードしたい run の ID。 |
| `-p, --project` | アップロード先の Project。 |
| `-e, --entity` | 使用する Entity。 |
| `--job_type` | 関連する run をまとめるための種類を指定します。 |
| `--sync-tensorboard / --no-sync-tensorboard` | TensorBoard の tfevent ファイルを wandb にストリームします。 |
| `--include-globs` | 含めるファイルをカンマ区切りで指定します。 |
| `--exclude-globs` | 除外するファイルをカンマ区切りで指定します。 |
| `--include-online / --no-include-online` | オンライン run を含める |
| `--include-offline / --no-include-offline` | オフライン run を含める |
| `--include-synced / --no-include-synced` | 同期済みの run を含める |
| `--mark-synced / --no-mark-synced` | run を同期済みとしてマークする |
| `--sync-all` | すべての run を同期する |
| `--clean` | 同期済みの run を削除する |
| `--clean-old-hours` | 指定した時間より前に作成された run を削除します（--clean フラグと併用）。 |
| `--clean-force` | 確認プロンプトなしでクリーン実行。 |
| `--show` | 表示する run の数 |
| `--append` | run を追加する |
| `--skip-console` | コンソールログをスキップする |