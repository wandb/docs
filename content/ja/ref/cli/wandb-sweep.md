---
title: wandb sweep
---

**使用方法**

`wandb sweep [OPTIONS] CONFIG_YAML_OR_SWEEP_ID`

**概要**

ハイパーパラメーター探索を初期化します。機械学習モデルのコスト関数を最適化するハイパーパラメーターを、さまざまな組み合わせでテストして探索します。

**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| `-p, --project` | Sweep から作成された W&B run を送信する Project の名前。Project が指定されていない場合は、run は「Uncategorized」とラベル付けされた Project に送信されます。 |
| `-e, --entity` | Sweep により作成された W&B run を送信したい Entity（ユーザー名またはチーム名）。指定した Entity がすでに存在していることを確認してください。Entity を指定しない場合、run はデフォルトの Entity（通常はあなたのユーザー名）に送信されます。 |
| `--controller` | ローカルコントローラを実行します |
| `--verbose` | 詳細な出力を表示します |
| `--name` | Sweep の名前。指定がなければ Sweep ID が使用されます。 |
| `--program` | Sweep 用のプログラムを設定します |
| `--update` | 保留中の Sweep を更新します |
| `--stop` | Sweep を終了して新しく run を開始せず、現在進行中の run のみ完了させます。 |
| `--cancel` | Sweep をキャンセルして、進行中のすべての run を強制終了し、新しい run の開始も止めます。 |
| `--pause` | Sweep を一時停止し、新しい run の開始を一時的に止めます。 |
| `--resume` | Sweep を再開し、新しい run の実行を継続します。 |
| `--prior_run` | この Sweep に追加する既存の run の ID |
