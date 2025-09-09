---
title: wandb sweep
menu:
  reference:
    identifier: ja-ref-cli-wandb-sweep
---

**使用方法**

`wandb sweep [OPTIONS] CONFIG_YAML_OR_SWEEP_ID`

**概要**

ハイパーパラメーター探索を初期化します。さまざまな組み合わせを試して、機械学習 モデルのコスト関数を最適化するハイパーパラメーターを探索します。


**オプション**

| **Option** | **Description** |
| :--- | :--- |
| `-p, --project` | sweep から作成された W&B の run を送信するプロジェクト名。プロジェクトを指定しない場合、run は Uncategorized というラベルの付いたプロジェクトに送信されます。 |
| `-e, --entity` | sweep により作成された W&B の run を送信したい送信先のユーザー名またはチーム名。指定した entity がすでに存在することを確認してください。entity を指定しない場合、run は既定の entity（通常はあなたのユーザー名）に送信されます。 |
| `--controller` | ローカル コントローラを実行します |
| `--verbose` | 詳細な出力を表示します |
| `--name` | sweep の名前。名前を指定しない場合は sweep ID が使用されます。 |
| `--program` | sweep のプログラムを設定します |
| `--update` | 保留中の sweep を更新します |
| `--stop` | sweep を終了し、新しい run の起動を止め、現在実行中の run の終了を待ちます。 |
| `--cancel` | sweep をキャンセルし、実行中のすべての run を停止して、新しい run の起動を止めます。 |
| `--pause` | sweep を一時停止し、新しい run の起動を一時的に止めます。 |
| `--resume` | sweep を再開し、新しい run の起動を継続します。 |
| `--prior_run` | この sweep に追加する既存の run の ID |