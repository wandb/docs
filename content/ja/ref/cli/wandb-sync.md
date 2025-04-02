---
title: wandb sync
menu:
  reference:
    identifier: ja-ref-cli-wandb-sync
---

**使用方法**

`wandb sync [OPTIONS] [PATH]...`

**概要**

オフラインのトレーニング ディレクトリー を W&B にアップロードします。

**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| `--id` | アップロード先の run を指定します。 |
| `-p, --project` | アップロード先の project を指定します。 |
| `-e, --entity` | スコープする entity を指定します。 |
| `--job_type` | 関連する run をグループ化するための run の種類を指定します。 |
| `--sync-tensorboard / --no-sync-tensorboard` | tfevent ファイルを wandb にストリームします。 |
| `--include-globs` | 含める glob のカンマ区切りリスト。 |
| `--exclude-globs` | 除外する glob のカンマ区切りリスト。 |
| `--include-online / --no-include-online` | オンライン run を含めます。 |
| `--include-offline / --no-include-offline` | オフライン run を含めます。 |
| `--include-synced / --no-include-synced` | 同期済みの run を含めます。 |
| `--mark-synced / --no-mark-synced` | run を同期済みとしてマークします。 |
| `--sync-all` | すべての run を同期します。 |
| `--clean` | 同期済みの run を削除します。 |
| `--clean-old-hours` | 指定した時間より前に作成された run を削除します。--clean フラグと組み合わせて使用​​します。 |
| `--clean-force` | 確認プロンプトなしでクリーンします。 |
| `--show` | 表示する run の数 |
| `--append` | run を追加します。 |
| `--skip-console` | コンソール ログをスキップします。 |
