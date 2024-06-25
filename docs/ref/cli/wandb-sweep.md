
# wandb sweep

**使用方法**

`wandb sweep [OPTIONS] CONFIG_YAML_OR_SWEEP_ID`

**概要**

ハイパーパラメーター探索を初期化します。 機械学習モデルのコスト関数を最適化するハイパーパラメーターを、さまざまな組み合わせを試して検索します。

**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| -p, --project | sweep から作成された W&B runs を送信するプロジェクトの名前。プロジェクトが指定されていない場合、run は Uncategorized とラベル付けされたプロジェクトに送られます。 |
| -e, --entity | sweep によって作成された W&B runs を送信するユーザー名またはチーム名。指定した entity が既に存在することを確認してください。entity を指定しない場合、run はデフォルトの entity に送信されます。通常、これはあなたのユーザー名です。 |
| --controller | ローカルコントローラを実行 |
| --verbose | 詳細な出力を表示 |
| --name | sweep の名前。名前が指定されていない場合は、sweep ID が使用されます。 |
| --program | sweep プログラムを設定 |
| --update | 保留中の sweep を更新 |
| --stop | 新しい run の実行を停止し、現在実行中の run を完了するために sweep を終了します。 |
| --cancel | すべての実行中の run を終了し、新しい run の実行を停止するために sweep をキャンセルします。 |
| --pause | 新しい run の実行を一時的に停止するために sweep を一時停止します。 |
| --resume | 新しい run の実行を続けるために sweep を再開します。 |