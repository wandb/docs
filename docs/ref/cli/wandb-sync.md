
# wandb sync

**使用方法**

`wandb sync [OPTIONS] [PATH]...`

**概要**

オフライントレーニングディレクトリーをW&Bにアップロード

**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| --id | アップロードしたいrun。 |
| -p, --project | アップロードしたいproject。 |
| -e, --entity | スコープとするentity。 |
| --job_type | 関連するrunsをグルーピングするためのrunの種類を指定します。 |
| --sync-tensorboard / --no-sync-tensorboard | tfeventファイルをwandbにストリームします。 |
| --include-globs | 含めるグロブのカンマ区切りリスト。 |
| --exclude-globs | 除外するグロブのカンマ区切りリスト。 |
| --include-online / --no-include-online | オンラインrunを含む |
| --include-offline / --no-include-offline | オフラインrunを含む |
| --include-synced / --no-include-synced | 同期済みrunを含む |
| --mark-synced / --no-mark-synced | runを同期済みとしてマーク |
| --sync-all | すべてのrunを同期 |
| --clean | 同期済みrunを削除 |
| --clean-old-hours | 指定した時間前に作成されたrunを削除。このフラグは--cleanと併用します。 |
| --clean-force | 確認プロンプトなしでクリーン |
| --show | 表示するrunの数 |
| --append | runを追加 |
| --skip-console | コンソールログをスキップ |