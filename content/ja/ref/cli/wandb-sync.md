---
title: wandb sync
menu:
  reference:
    identifier: ja-ref-cli-wandb-sync
---

**使用方法**

`wandb sync [OPTIONS] [PATH]...`

**概要**

オフライン トレーニング ディレクトリーを W&B にアップロードします


**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| `--id` | アップロードしたい run。 |
| `-p, --project` | アップロードしたいプロジェクト。 |
| `-e, --entity` | スコープにするエンティティ。 |
| `--job_type` | 関連する runs をまとめる run のタイプを指定します。 |
| `--sync-tensorboard / --no-sync-tensorboard` | tfevent ファイルを wandb にストリームします。 |
| `--include-globs` | 含めるグロブのカンマ区切りのリスト。 |
| `--exclude-globs` | 除外するグロブのカンマ区切りのリスト。 |
| `--include-online / --no-include-online` | オンライン runs を含める |
| `--include-offline / --no-include-offline` | オフライン runs を含める |
| `--include-synced / --no-include-synced` | 同期済み runs を含める |
| `--mark-synced / --no-mark-synced` | runs を同期済みとしてマークする |
| `--sync-all` | 全ての runs を同期する |
| `--clean` | 同期済み runs を削除する |
| `--clean-old-hours` | 指定した時間より前に作成された runs を削除します。--clean フラグと一緒に使用します。 |
| `--clean-force` | 確認プロンプトなしでクリーンする。 |
| `--show` | 表示する runs の数 |
| `--append` | run を追加する |
| `--skip-console` | コンソールログをスキップする |