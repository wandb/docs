---
title: wandb sweep
menu:
  reference:
    identifier: ja-ref-cli-wandb-sweep
---

**使用方法**

`wandb sweep [OPTIONS] CONFIG_YAML_OR_SWEEP_ID`

**概要**

ハイパーパラメーター探索を初期化します。機械学習モデルのコスト関数を最適化するハイパーパラメーターを、さまざまな組み合わせをテストすることで探索します。

**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| `-p, --project` | Sweep から作成される W&B Run を送信する Project の名前を指定します。Project が指定されていない場合、Run は「Uncategorized」という Project に送信されます。 |
| `-e, --entity` | Sweep によって作成される W&B Run を送信したいユーザー名またはチーム名を指定します。指定した Entity が既に存在していることを確認してください。指定しない場合は、デフォルトの Entity（通常は自分のユーザー名）に送信されます。 |
| `--controller` | ローカルのコントローラを実行します |
| `--verbose` | 詳細な出力を表示します |
| `--name` | Sweep の名前を指定します。名前を指定しない場合は Sweep ID が使用されます。 |
| `--program` | Sweep で実行するプログラムを設定します |
| `--update` | 保留中の Sweep を更新します |
| `--stop` | Sweep を完了して新たな Run の実行を停止し、現在実行中の Run はそのまま終了させます。 |
| `--cancel` | Sweep をキャンセルしてすべての実行中の Run を停止し、新たな Run の実行も停止します。 |
| `--pause` | Sweep を一時停止し、新たな Run の実行を一時的に止めます。 |
| `--resume` | Sweep を再開して新しい Run の実行を続けます。 |
| `--prior_run` | この Sweep に追加する既存の Run の ID を指定します |