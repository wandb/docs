---
title: 登録済みモデルを作成する
description: 登録済みモデルを作成し、モデリングタスクの候補モデルをすべてまとめましょう。
menu:
  default:
    identifier: ja-guides-core-registry-model_registry-create-registered-model
    parent: model-registry
weight: 4
---

[**registered model**]({{< relref path="./model-management-concepts.md#registered-model" lang="ja" >}}) を作成して、モデリングタスクの候補モデルをまとめましょう。registered model は Model Registry 内で対話的に作成することも、Python SDK を使ってプログラムから作成することもできます。

## プログラムから registered model を作成する

W&B Python SDK を使用して model を登録できます。指定した registered model が存在しない場合、W&B が自動的に registered model を作成します。

`<>` で囲まれた値は、ご自身の設定に置き換えてください。

```python
import wandb

run = wandb.init(entity="<entity>", project="<project>")
run.link_model(path="<path-to-model>", registered_model_name="<registered-model-name>")
run.finish()
```

`registered_model_name` で指定する名前は [Model Registry App](https://wandb.ai/registry/model) に表示されます。

## 対話的に registered model を作成する

[Model Registry App](https://wandb.ai/registry/model) で対話的に registered model を作成します。

1. [Model Registry App](https://wandb.ai/registry/model) にアクセスします。
{{< img src="/images/models/create_registered_model_1.png" alt="Model Registry のトップページ" >}}
2. Model Registry ページ右上にある **New registered model** ボタンをクリックします。
{{< img src="/images/models/create_registered_model_model_reg_app.png" alt="New registered model ボタン" >}}
3. 表示されたパネルで、**Owning Entity** のドロップダウンから registered model を紐付けたい Entity を選択します。
{{< img src="/images/models/create_registered_model_3.png" alt="モデル作成フォーム" >}}
4. **Name** 欄に model の名前を入力します。
5. **Type** のドロップダウンから、registered model にリンクしたいアーティファクトの種類を選びます。
6. （任意）**Description** 欄に model について説明を記入します。
7. （任意）**Tags** 欄に 1 つ以上のタグを追加できます。
8. **Register model** をクリックします。

{{% alert %}}
model を model registry に手動でリンクするのは、単発のモデル管理には便利です。ただし、多くの場合は [プログラムから model バージョンを model registry にリンクする]({{< relref path="link-model-version#programmatically-link-a-model" lang="ja" >}}) 方法が役立ちます。

たとえば、毎晩自動で実行するジョブがある場合、毎晩作成される model を都度手動でリンクするのは手間です。代わりに、model の評価とパフォーマンス向上時のみ model registry へリンクするスクリプトを作成し、W&B Python SDK で自動化するのが便利です。
{{% /alert %}}