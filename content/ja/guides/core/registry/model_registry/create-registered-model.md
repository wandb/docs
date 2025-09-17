---
title: Registered Models を作成する
description: モデリングタスクのすべての候補モデルを管理するために、Registered Model を作成します。
menu:
  default:
    identifier: ja-guides-core-registry-model_registry-create-registered-model
    parent: model-registry
weight: 4
---

モデリング タスクのすべての候補モデルを保持する [Registered Model]({{< relref path="./model-management-concepts.md#registered-model" lang="ja" >}}) を作成します。Registered Model は、モデルレジストリ内でインタラクティブに作成することも、W&B Python SDK を使用してプログラムで作成することもできます。

## プログラムで Registered Model を作成する
W&B Python SDK を使用してプログラムでモデルを登録します。Registered Model が存在しない場合、W&B は自動的に Registered Model を作成します。

`<` `>` で囲まれた他の値を自分の値に置き換えてください:

```python
import wandb

run = wandb.init(entity="<entity>", project="<project>")
run.link_model(path="<path-to-model>", registered_model_name="<registered-model-name>")
run.finish()
```

`registered_model_name` に指定した名前は、[Model Registry App](https://wandb.ai/registry/model) に表示される名前です。

## インタラクティブに Registered Model を作成する
[Model Registry App](https://wandb.ai/registry/model) 内でインタラクティブに Registered Model を作成します。

1. [Model Registry App](https://wandb.ai/registry/model) に移動します。
{{< img src="/images/models/create_registered_model_1.png" alt="モデルレジストリのランディングページ" >}}
2. Model Registry ページの右上にある **New Registered Model** ボタンをクリックします。
{{< img src="/images/models/create_registered_model_model_reg_app.png" alt="New Registered Model ボタン" >}}
3. 表示されるパネルから、**Owning Entity** ドロップダウンで Registered Model を所属させたい Entities を選択します。
{{< img src="/images/models/create_registered_model_3.png" alt="モデル作成フォーム" >}}
4. **Name** フィールドに Registered Model の名前を入力します。
5. **Type** ドロップダウンから、Registered Model にリンクする Artifacts のタイプを選択します。
6. (オプション) **Description** フィールドに Registered Model の説明を追加します。
7. (オプション) **Tags** フィールドに 1 つ以上のタグを追加します。
8. **Register model** をクリックします。

{{% alert %}}
モデルをモデルレジストリに手動でリンクすることは、使い捨てのモデルには役立ちます。ただし、[モデルのバージョンをモデルレジストリにプログラムでリンクする]({{< relref path="link-model-version#programmatically-link-a-model" lang="ja" >}}) こともよくあります。

たとえば夜間ジョブがあるとします。毎晩作成されるモデルを手動でリンクするのは面倒です。代わりに、モデルを評価するスクリプトを作成し、モデルのパフォーマンスが向上した場合、そのモデルを W&B Python SDK でモデルレジストリにリンクできます。
{{% /alert %}}