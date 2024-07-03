# wandb sweep

**使用方法**

`wandb sweep [OPTIONS] CONFIG_YAML_OR_SWEEP_ID`

**概要**

ハイパーパラメーター探索を初期化します。機械学習モデルのコスト関数を最適化するハイパーパラメーターを、様々な組み合わせをテストして探索します。

**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| -p, --project | sweep から作成される W&B runs が送信されるプロジェクトの名前。プロジェクトが指定されていない場合、run は Uncategorized というラベルのプロジェクトに送信されます。 |
| -e, --entity | sweep によって作成された W&B runs を送信したいユーザー名またはチーム名。このエンティティが既に存在することを確認してください。エンティティを指定しない場合、run はデフォルトのエンティティ（通常はユーザーネーム）に送信されます。 |
| --controller | ローカルコントローラを実行する |
| --verbose | 詳細な出力を表示する |
| --name | sweep の名前。名前が指定されていない場合、sweep ID が使用されます。 |
| --program | sweep プログラムを設定 |
| --update | 保留中の sweep を更新 |
| --stop | 新しい runs の実行を停止して、現在実行中の runs を完了させるために sweep を終了 |
| --cancel | 実行中のすべての runs を終了し、新しい runs の実行を停止するために sweep をキャンセル |
| --pause | 一時的に新しい runs の実行を停止するために sweep を一時停止 |
| --resume | 新しい runs の実行を続行するために sweep を再開 |
| --prior_run | この sweep に追加する既存の run の ID |