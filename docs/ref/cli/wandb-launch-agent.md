
# wandb launch-agent

**使用方法**

`wandb launch-agent [OPTIONS]`

**概要**

W&B launch エージェントを実行します。

**オプション**

| **Option** | **Description** |
| :--- | :--- |
| -q, --queue <queue(s)> | エージェントが監視するキューの名前。複数の -q フラグがサポートされています。 |
| -e, --entity | 使用する entity。デフォルトは現在ログイン中のユーザーです。 |
| -l, --log-file | 内部エージェントログの保存先。- を使用してstdoutに出力。デフォルトでは、すべてのエージェントログは wandb/ サブディレクトリーまたは設定されている場合 WANDB_DIR のdebug.logに出力されます。 |
| -j, --max-jobs | このエージェントが並行して実行できる最大の launch ジョブ数。デフォルトは1。上限なしにするには -1 を設定します。 |
| -c, --config | 使用するエージェントの config yaml のパス |
| -v, --verbose | 詳細な出力を表示 |
