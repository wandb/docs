---
title: 登録済みモデルを作成する
description: モデル作成のタスクのために、すべての候補モデルを保持する Registered Model を作成します。
menu:
  default:
    identifier: ja-guides-core-registry-model_registry-create-registered-model
    parent: model-registry
weight: 4
---

[registered model]({{< relref path="./model-management-concepts.md#registered-model" lang="ja" >}}) を作成し、モデリングタスクのすべての候補モデルを保持します。モデルレジストリ内でインタラクティブに、または Python SDK を使用してプログラム的に registered model を作成できます。

## プログラムで registered model を作成する

W&B Python SDK を使用してモデルを登録します。registered model が存在しない場合、W&B は自動的に registered model を作成します。

`<>` で囲まれた他の値をあなた自身のもので置き換えてください:

```python
import wandb

run = wandb.init(entity="<entity>", project="<project>")
run.link_model(path="<path-to-model>", registered_model_name="<registered-model-name>")
run.finish()
```

`registered_model_name` に指定した名前は [Model Registry App](https://wandb.ai/registry/model) に表示される名前です。

## インタラクティブに registered model を作成する

[Model Registry App](https://wandb.ai/registry/model) でインタラクティブに registered model を作成します。

1. Model Registry App に移動します: [https://wandb.ai/registry/model](https://wandb.ai/registry/model)。
{{< img src="/images/models/create_registered_model_1.png" alt="" >}}
2. Model Registry ページの右上にある **New registered model** ボタンをクリックします。
{{< img src="/images/models/create_registered_model_model_reg_app.png" alt="" >}}
3. 表示されたパネルから、registered model が属するエンティティを **Owning Entity** ドロップダウンから選択します。
{{< img src="/images/models/create_registered_model_3.png" alt="" >}}
4. **Name** フィールドにモデルの名前を入力します。 
5. **Type** ドロップダウンから、registered model とリンクするアーティファクトのタイプを選択します。
6. (オプション) **Description** フィールドにモデルについての説明を追加します。
7. (オプション) **Tags** フィールドに1つ以上のタグを追加します。
8. **Register model** をクリックします。

{{% alert %}}
モデルをモデルレジストリに手動でリンクすることは、一度だけのモデルに便利です。しかし、[プログラムでモデルバージョンをモデルレジストリにリンクする]({{< relref path="link-model-version#programmatically-link-a-model" lang="ja" >}})こともよくあります。

例えば、毎晩のジョブがあるとします。毎晩作成されるモデルを手動でリンクするのは面倒です。代わりに、モデルを評価し、そのモデルがパフォーマンスを改善した場合にそのモデルを W&B Python SDK を使用してモデルレジストリにリンクするスクリプトを作成することができます。
{{% /alert %}}