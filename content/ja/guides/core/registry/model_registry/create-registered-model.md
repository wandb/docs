---
title: Create a registered model
description: モデリングタスクのすべての候補モデルを保持するために、登録済み モデル を作成します。
menu:
  default:
    identifier: ja-guides-core-registry-model_registry-create-registered-model
    parent: model-registry
weight: 4
---

モデリングタスクのすべての候補モデルを保持するために、[registered model]({{< relref path="./model-management-concepts.md#registered-model" lang="ja" >}})を作成します。 registered model は、Model Registry 内でインタラクティブに、または Python SDK でプログラム的に作成できます。

## プログラムで registered model を作成する
W&B Python SDK でモデルをプログラム的に登録します。 registered model が存在しない場合、W&B は自動的に registered model を作成します。

必ず、`<>` で囲まれた値を独自の値に置き換えてください。

```python
import wandb

run = wandb.init(entity="<entity>", project="<project>")
run.link_model(path="<path-to-model>", registered_model_name="<registered-model-name>")
run.finish()
```

`registered_model_name` に指定した名前は、[Model Registry App](https://wandb.ai/registry/model) に表示される名前です。

## インタラクティブに registered model を作成する
[Model Registry App](https://wandb.ai/registry/model) 内でインタラクティブに registered model を作成します。

1. [https://wandb.ai/registry/model](https://wandb.ai/registry/model) で Model Registry App に移動します。
{{< img src="/images/models/create_registered_model_1.png" alt="" >}}
2. Model Registry ページの右上にある **New registered model** ボタンをクリックします。
{{< img src="/images/models/create_registered_model_model_reg_app.png" alt="" >}}
3. 表示される パネル で、registered model が属する Entities を **Owning Entity** ドロップダウンから選択します。
{{< img src="/images/models/create_registered_model_3.png" alt="" >}}
4. **Name** フィールドにモデルの名前を入力します。
5. **Type** ドロップダウンから、registered model にリンクする Artifacts のタイプを選択します。
6. （オプション）**Description** フィールドにモデルに関する説明を追加します。
7. （オプション）**Tags** フィールドに、1 つ以上のタグを追加します。
8. **Register model** をクリックします。

{{% alert %}}
モデルを Model Registry に手動でリンクすることは、1 回限りのモデルに役立ちます。ただし、多くの場合、[プログラムでモデルバージョンを Model Registry にリンクする]({{< relref path="link-model-version#programmatically-link-a-model" lang="ja" >}})ことが役立ちます。

たとえば、毎晩のジョブがあるとします。毎晩作成されるモデルを手動でリンクするのは面倒です。代わりに、モデルを評価するスクリプトを作成し、モデルのパフォーマンスが向上した場合、W&B Python SDK を使用してそのモデルを Model Registry にリンクすることができます。
{{% /alert %}}
