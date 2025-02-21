---
title: Create a registered model
description: モデリングタスクのすべての候補モデルを保持するために、登録済み モデル を作成します。
menu:
  default:
    identifier: ja-guides-models-registry-model_registry-create-registered-model
    parent: model-registry
weight: 4
---

モデリングタスクのすべての候補モデルを保持するために、[登録済みモデル]({{< relref path="./model-management-concepts.md#registered-model" lang="ja" >}})を作成します。登録済みモデルは、モデルレジストリ内でインタラクティブに作成するか、Python SDK でプログラム的に作成できます。

## プログラム的に登録済みモデルを作成する
W&B Python SDK を使用して、プログラム的にモデルを登録します。登録済みモデルが存在しない場合、W&B は自動的に登録済みモデルを作成します。

`< >` で囲まれた値を必ず独自の値に置き換えてください。

```python
import wandb

run = wandb.init(entity="<entity>", project="<project>")
run.link_model(path="<path-to-model>", registered_model_name="<registered-model-name>")
run.finish()
```

`registered_model_name` に指定した名前が、[Model Registry App](https://wandb.ai/registry/model) に表示される名前になります。

## インタラクティブに登録済みモデルを作成する
[Model Registry App](https://wandb.ai/registry/model) 内でインタラクティブに登録済みモデルを作成します。

1. [https://wandb.ai/registry/model](https://wandb.ai/registry/model) の Model Registry App に移動します。
{{< img src="/images/models/create_registered_model_1.png" alt="" >}}
2. Model Registry ページの右上にある [**New registered model**] ボタンをクリックします。
{{< img src="/images/models/create_registered_model_model_reg_app.png" alt="" >}}
3. 表示される パネル で、登録済みモデルが属する Entity を [**Owning Entity**] ドロップダウンから選択します。
{{< img src="/images/models/create_registered_model_3.png" alt="" >}}
4. [**Name**] フィールドにモデルの名前を入力します。
5. [**Type**] ドロップダウンから、登録済みモデルにリンクする Artifacts のタイプを選択します。
6. （オプション）[**Description**] フィールドにモデルの説明を追加します。
7. （オプション）[**Tags**] フィールド内で、1 つまたは複数のタグを追加します。
8. [**Register model**] をクリックします。

{{% alert %}}
モデルをモデルレジストリに手動でリンクすることは、単発のモデルに役立ちます。ただし、多くの場合、[モデル バージョンをモデルレジストリにプログラム的にリンクする]({{< relref path="link-model-version#programmatically-link-a-model" lang="ja" >}})と便利です。

たとえば、毎晩実行するジョブがあるとします。毎晩作成されたモデルを手動でリンクするのは面倒です。代わりに、モデルを評価するスクリプトを作成し、モデルのパフォーマンスが向上した場合、W&B Python SDK を使用してそのモデルをモデルレジストリにリンクすることができます。
{{% /alert %}}
