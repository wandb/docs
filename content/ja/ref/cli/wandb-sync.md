---
title: wandb sync
menu:
  reference:
    identifier: ja-ref-cli-wandb-sync
---

**使い方**

`wandb sync [OPTIONS] [PATH]...`

**概要**

オフラインのトレーニング ディレクトリーを W&B にアップロードします。

**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| `--id` | アップロードしたい run の ID を指定します。 |
| `-p, --project` | アップロード先の Project を指定します。 |
| `-e, --entity` | 対象とする Entity を指定します。 |
| `--job_type` | run のタイプを指定し、関連する run をまとめて管理します。 |
| `--sync-tensorboard / --no-sync-tensorboard` | tfevent ファイルを wandb にストリームします。 |
| `--include-globs` | インクルードするファイルのパターンをカンマ区切りで指定します。 |
| `--exclude-globs` | 除外するファイルのパターンをカンマ区切りで指定します。 |
| `--include-online / --no-include-online` | オンライン run を含めます。 |
| `--include-offline / --no-include-offline` | オフライン run を含めます。 |
| `--include-synced / --no-include-synced` | 同期済みの run を含めます。 |
| `--mark-synced / --no-mark-synced` | run を同期済みにマークします。 |
| `--sync-all` | すべての run を同期します。 |
| `--clean` | 同期済みの run を削除します。 |
| `--clean-old-hours` | 指定した時間より前に作成された run を削除します（--clean オプションと併用）。 |
| `--clean-force` | 確認プロンプトなしでクリーンアップを実行します。 |
| `--show` | 表示する run の数を指定します。 |
| `--append` | run を追加します。 |
| `--skip-console` | コンソールログをスキップします。 |