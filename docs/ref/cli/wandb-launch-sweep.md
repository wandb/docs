# wandb launch-sweep

**使用法**

`wandb launch-sweep [OPTIONS] [CONFIG]`

**概要**

W&Bのローンンチsweepを実行します（実験的機能）。

**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| -q, --queue | sweepを投入するキューの名前 |
| -p, --project | エージェントが監視するプロジェクトの名前。指定されると、設定ファイルを使用して渡されるプロジェクトの値を上書きします |
| -e, --entity | 使用するエンティティ。デフォルトは現在ログイン中のユーザーです |
| -r, --resume_id | 8文字のsweep IDを渡してローンチsweepを再開します。キューが必要です |
| --prior_run | このsweepに追加する既存のrunのID |