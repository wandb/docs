---
title: wandb sweep
menu:
  reference:
    identifier: ja-ref-cli-wandb-sweep
---

**使用法**

`wandb sweep [OPTIONS] CONFIG_YAML_OR_SWEEP_ID`

**概要**

ハイパーパラメーター探索を初期化します。機械学習モデルのコスト関数を最適化するために、さまざまな組み合わせをテストしてハイパーパラメーターを検索します。

**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| `-p, --project` | sweep から作成された W&B によって送信されるプロジェクトの名前です。プロジェクトが指定されない場合、run は Uncategorized というラベルのプロジェクトに送信されます。 |
| `-e, --entity` | sweep によって作成された W&B run を送信したいユーザー名またはチーム名です。指定した entity が既に存在することを確認してください。entity を指定しない場合、run は通常は自分のユーザー名であるデフォルトの entity に送信されます。 |
| `--controller` | ローカルコントローラを実行します |
| `--verbose` | 詳細情報を表示します |
| `--name` | sweep の名前です。名前が指定されていない場合は、sweep ID が使用されます。 |
| `--program` | sweep プログラムを設定します |
| `--update` | 保留中の sweep を更新します |
| `--stop` | 新しい run の実行を停止して、現在実行中の run が終了するように sweep を終了します。 |
| `--cancel` | 実行中のすべての run を停止し、新しい run の実行を停止するために sweep をキャンセルします。 |
| `--pause` | 新しい run の実行を一時的に停止するために sweep を一時停止します。 |
| `--resume` | 新しい run の実行を再開するために sweep を再開します。 |
| `--prior_run` | この sweep に追加する既存の run の ID |
