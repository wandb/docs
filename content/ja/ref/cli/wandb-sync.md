---
title: wandb sync
menu:
  reference:
    identifier: ja-ref-cli-wandb-sync
---

**使い方**

`wandb sync [OPTIONS] [PATH]...`

**概要**

オフラインのトレーニング ディレクトリーを W&B にアップロード


**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| `--id` | アップロード先の run。 |
| `-p, --project` | アップロード先の Project。 |
| `-e, --entity` | スコープ対象の Entity。 |
| `--job_type` | 関連する runs をまとめるための run の種類を指定。 |
| `--sync-tensorboard / --no-sync-tensorboard` | tfevent ファイルを W&B にストリーミング。 |
| `--include-globs` | 含める glob パターンのカンマ区切りリスト。 |
| `--exclude-globs` | 除外する glob パターンのカンマ区切りリスト。 |
| `--include-online / --no-include-online` | オンラインの runs を含める。 |
| `--include-offline / --no-include-offline` | オフラインの runs を含める。 |
| `--include-synced / --no-include-synced` | 同期済みの runs を含める。 |
| `--mark-synced / --no-mark-synced` | runs を同期済みとしてマーク。 |
| `--sync-all` | すべての runs を同期。 |
| `--clean` | 同期済みの runs を削除。 |
| `--clean-old-hours` | 指定時間より前に作成された runs を削除。--clean フラグと併用。 |
| `--clean-force` | 確認プロンプトなしでクリーンアップ。 |
| `--show` | 表示する runs の数。 |
| `--append` | run を追記。 |
| `--skip-console` | コンソール ログをスキップ。 |
| `--replace-tags` | 'old_tag1=new_tag1,old_tag2=new_tag2' の形式でタグを置換。 |