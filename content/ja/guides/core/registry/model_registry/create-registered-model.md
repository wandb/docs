---
title: 登録済みモデルを作成する
description: 登録済みモデルを作成して、モデリングタスクのための候補モデルをすべて管理しましょう。
menu:
  default:
    identifier: create-registered-model
    parent: model-registry
weight: 4
---

モデリングタスクの全ての候補モデルをまとめるために、[registered model]({{< relref "./model-management-concepts.md#registered-model" >}}) を作成しましょう。registered model は Model Registry 内でインタラクティブに、または Python SDK からプログラムで作成できます。

## プログラムから registered model を作成する

W&B Python SDK を使ってプログラムからモデルを登録しましょう。registered model が存在しない場合は、W&B が自動的に registered model を作成します。

`<>` で囲まれている値は、ご自身のものに置き換えてください。

```python
import wandb

# Entity と Project を指定して run を初期化
run = wandb.init(entity="<entity>", project="<project>")
# モデルを登録済みモデルにリンク
run.link_model(path="<path-to-model>", registered_model_name="<registered-model-name>")
run.finish()
```

`registered_model_name` に指定する名前が [Model Registry App](https://wandb.ai/registry/model) に表示される registered model の名前になります。

## インタラクティブに registered model を作成する

[Model Registry App](https://wandb.ai/registry/model) でインタラクティブに registered model を作成することもできます。

1. [Model Registry App](https://wandb.ai/registry/model) にアクセスします。
{{< img src="/images/models/create_registered_model_1.png" alt="Model Registry landing page" >}}
2. Model Registry ページ右上の **New registered model** ボタンをクリックします。
{{< img src="/images/models/create_registered_model_model_reg_app.png" alt="New registered model button" >}}
3. 表示されるパネルで、**Owning Entity** ドロップダウンから registered model を紐付けたい Entity を選択します。
{{< img src="/images/models/create_registered_model_3.png" alt="Model creation form" >}}
4. **Name** フィールドにモデルの名前を入力します。 
5. **Type** ドロップダウンで、registered model にリンクする artifacts のタイプを選択します。
6. （任意）**Description** フィールドにモデルについての説明を追加します。
7. （任意）**Tags** フィールドでタグを一つ以上追加します。 
8. **Register model** をクリックします。

{{% alert %}}
手動でモデルを model registry にリンクするのは、一度きりのモデルには便利です。しかし、[プログラムからモデルの version を model registry にリンクする]({{< relref "link-model-version#programmatically-link-a-model" >}}) のが役立つ場面も多くあります。

例えばナイトリービルドのジョブがある場合、毎晩作成されるモデルを手動で model registry にリンクするのは手間がかかります。その代わりに、モデルを評価して性能が向上している場合のみ、そのモデルを W&B Python SDK で model registry にリンクするスクリプトを作成することができます。
{{% /alert %}}