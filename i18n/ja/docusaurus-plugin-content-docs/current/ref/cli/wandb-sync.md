# wandb sync

**使い方**

`wandb sync [オプション] [パス]...`

**概要**

オフラインのトレーニングディレクトリをW&Bにアップロードします。

**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| --id | アップロードするrunのID |
| -p, --project | アップロードするプロジェクト |
| -e, --entity | スコープするエンティティ |
| --sync-tensorboard / --no-sync-tensorboard | tfeventファイルをwandbにストリーム |
| --include-globs | インクルードするglobのカンマ区切りリスト |
| --exclude-globs | 除外するglobのカンマ区切りリスト |
| --include-online / --no-include-online | オンラインのrunを含める |
| --include-offline / --no-include-offline | オフラインのrunを含める |
| --include-synced / --no-include-synced | 同期済みのrunを含める |
| --mark-synced / --no-mark-synced | runを同期済みとマークする |
| --sync-all | すべてのrunを同期する |
| --clean | 同期済みのrunを削除する |
| --clean-old-hours | 指定した時間より前に作成されたrunを削除。`--clean`フラグと一緒に使用する |
| --clean-force | 確認プロンプトなしでクリーニング |
| --show | 表示するrunの数 |
| --append | runに追加する |
| --help | このメッセージを表示して終了 |