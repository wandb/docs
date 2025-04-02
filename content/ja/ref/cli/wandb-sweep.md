---
title: wandb sweep
menu:
  reference:
    identifier: ja-ref-cli-wandb-sweep
---

**Usage**

`wandb sweep [OPTIONS] CONFIG_YAML_OR_SWEEP_ID`

**Summary**

ハイパーパラメータ ー探索を初期化します。様々な組み合わせをテストすることで、機械学習 モデルのコスト関数を最適化するハイパーパラメータ ーを探索します。

**Options**

| **Option** | **Description** |
| :--- | :--- |
| `-p, --project` | sweep から作成された W&B の run が送信される project の名前。 project が指定されていない場合、run は「Uncategorized（未分類）」というラベルの付いた project に送信されます。 |
| `-e, --entity` | sweep によって作成された W&B の run を送信したい user 名または Team 名。指定する Entity がすでに存在することを確認してください。 Entity を指定しない場合、run はデフォルトの Entity（通常は user 名）に送信されます。 |
| `--controller` | ローカルのコントローラ を実行 |
| `--verbose` | 詳細な出力を表示 |
| `--name` | sweep の名前。名前が指定されていない場合は、sweep ID が使用されます。 |
| `--program` | sweep プログラムを設定 |
| `--update` | 保留中の sweep を更新 |
| `--stop` | sweep を終了して、新しい run の実行を停止し、現在実行中の run を完了させます。 |
| `--cancel` | sweep をキャンセルして、実行中のすべての run を強制終了し、新しい run の実行を停止します。 |
| `--pause` | sweep を一時停止して、新しい run の実行を一時的に停止します。 |
| `--resume` | sweep を再開して、新しい run の実行を継続します。 |
| `--prior_run` | この sweep に追加する既存の run の ID |
