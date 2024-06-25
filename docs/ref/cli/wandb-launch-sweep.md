
# wandb launch-sweep

**使用方法**

`wandb launch-sweep [OPTIONS] [CONFIG]`

**概要**

W&B の launch sweep を実行します（実験的機能）。

**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| -q, --queue | sweep をプッシュするためのキューの名前 |
| -p, --project | エージェントが監視するプロジェクトの名前。指定された場合、設定ファイルで渡された project 値を上書きします |
| -e, --entity | 使用する entity。デフォルトは現在ログイン中のユーザー |
| -r, --resume_id | 8文字の sweep ID を渡して launch sweep を再開します。キューが必要です |