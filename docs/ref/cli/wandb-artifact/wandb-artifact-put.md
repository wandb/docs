
# wandb artifact put

**使用方法**

`wandb artifact put [OPTIONS] PATH`

**概要**

artifact を wandb にアップロードする

**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| -n, --name | プッシュするアーティファクトの名前: Projects/artifact_name |
| -d, --description | このアーティファクトの説明 |
| -t, --type | アーティファクトの種類 |
| -a, --alias | このアーティファクトに適用するエイリアス |
| --id | アップロードしたい run |
| --resume | 現在のディレクトリーから最後の run を再開する |
| --skip_cache | アーティファクトファイルのアップロード中にキャッシュをスキップする |
| --policy [mutable|immutable] | アーティファクトファイルのアップロード時にストレージポリシーを設定する |