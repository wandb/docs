---
title: モデル バージョンをリンクする
description: W&B App または Python SDK を使って、 model バージョン を Registered Models にリンクできます。
menu:
  default:
    identifier: link-model-version
    parent: model-registry
weight: 5
---

W&B App または Python SDK を使用して、モデル バージョンを Registered Model にリンクできます。

## プログラムでモデルをリンクする

[`link_model`]({{< relref "/ref/python/sdk/classes/run.md#link_model" >}}) メソッドを使って、モデルファイルを W&B run にログし、[W&B Model Registry]({{< relref "./" >}}) にプログラムでリンクできます。

`<>` で囲まれている値は、ご自身の環境に合わせて置き換えてください。

```python
import wandb

run = wandb.init(entity="<entity>", project="<project>")
run.link_model(path="<path-to-model>", registered_model_name="<registered-model-name>")
run.finish()
```

もし指定した `registered-model-name` パラメータの名前の Registered Model がまだ存在しない場合、W&B が新しく Registered Model を作成します。

たとえば、「Fine-Tuned-Review-Autocompletion」（`registered-model-name="Fine-Tuned-Review-Autocompletion"`）という既存の Registered Model があり、すでにいくつかのモデルバージョン `v0`, `v1`, `v2` がリンクされているとします。この状態で同じ Registered Model 名（`registered-model-name="Fine-Tuned-Review-Autocompletion"`）を使って新しいモデルをリンクすると、W&B はこのモデルを既存の Registered Model に紐付け、新しいモデルバージョン `v3` を割り当てます。この名前の Registered Model が存在しなければ、新たに作成されバージョンは `v0` になります。

["Fine-Tuned-Review-Autocompletion" の Registered Model の例はこちら](https://wandb.ai/reviewco/registry/model?selectionPath=reviewco%2Fmodel-registry%2FFinetuned-Review-Autocompletion&view=all-models) で確認できます。

## インタラクティブにモデルをリンクする

Model Registry または Artifact ブラウザから、モデルをインタラクティブにリンクできます。

{{< tabpane text=true >}}
  {{% tab header="Model Registry" %}}
1. [Model Registry App](https://wandb.ai/registry/model) にアクセスします。
2. 新しいモデルをリンクしたい Registered Model 名の横にマウスを移動します。
3. **View details** の横にあるミートボールメニュー（3つの横点）アイコンをクリックします。
4. ドロップダウンから **Link new version** を選択します。
5. **Project** ドロップダウンから、該当するモデルが含まれるプロジェクト名を選択します。
6. **Model Artifact** ドロップダウンから、登録したいモデルアーティファクト名を選択します。
7. **Version** ドロップダウンで、Registered Model にリンクしたいモデルバージョンを選びます。

{{< img src="/images/models/link_model_wmodel_reg.gif" alt="モデルバージョンをレジストリにリンクする操作" >}}
  {{% /tab %}}
  {{% tab header="Artifact browser" %}}
1. W&B App でプロジェクトの Artifact ブラウザにアクセスします: `https://wandb.ai/<entity>/<project>/artifacts`
2. 左サイドバーの Artifacts アイコンをクリックします。
3. Registry にリンクしたいモデルバージョンをクリックします。
4. **Version overview** セクションで、**Link to registry** ボタンをクリックします。
5. 画面右側に表示されたモーダルの **Select a register model** メニューから、Registered Model を選びます。
6. **Next step** をクリックします。
7. （任意）**Aliases** ドロップダウンからエイリアスを選択します。
8. **Link to registry** をクリックします。

{{< img src="/images/models/manual_linking.gif" alt="手動でのモデルリンク操作" >}}  
  {{% /tab %}}
{{< /tabpane >}}

## リンクされたモデルのソースを表示する

リンクされたモデルのソースを表示する方法は2つあります。モデルがログされているプロジェクト内の Artifact ブラウザか、W&B Model Registry のどちらかです。

ポインタが、Model Registry の特定のモデルバージョンと、ソースとなるモデル Artifact（モデルがログされているプロジェクト内）を相互に結び付けます。ソースのモデル Artifact にも Model Registry へのポインタが表示されます。

{{< tabpane text=true >}}
  {{% tab header="Model Registry" %}}
1. [Model Registry App](https://wandb.ai/registry/model) にアクセスします。
{{< img src="/images/models/create_registered_model_1.png" alt="Registered Model の作成" >}}
2. Registered Model 名の横にある **View details** をクリックします。
3. **Versions** セクションで、調べたいモデルバージョンの横にある **View** を選びます。
4. 右側パネルの **Version** タブをクリックします。
5. **Version overview** セクションに **Source Version** という行があります。ここに、モデル名とそのバージョン（例：`mnist_model:v0`）が表示されます。

以下の画像では、`mnist_model` という `v0` モデルバージョンが（**Source version** フィールドの `mnist_model:v0` 参照）、`MNIST-dev` という Registered Model にリンクされています。

{{< img src="/images/models/view_linked_model_registry.png" alt="レジストリ上のリンクモデル" >}}  
  {{% /tab %}}
  {{% tab header="Artifact browser" %}}
1. W&B App でプロジェクトの Artifact ブラウザにアクセスします: `https://wandb.ai/<entity>/<project>/artifacts`
2. 左サイドバーの Artifacts アイコンを選択します。
3. Artifacts パネルから **model** のドロップダウンメニューを展開します。
4. Model Registry にリンク済みのモデル名とバージョンを選択します。
5. 右パネルの **Version** タブをクリックします。
6. **Version overview** セクションに **Linked To** という行があります。ここに Registered Model 名とバージョン(`registered-model-name:version`)が表示されます。

以下の画像では、`MNIST-dev` という Registered Model（**Linked To** フィールド参照）があり、`mnist_model` というモデルバージョン `v0`（`mnist_model:v0`）が、この `MNIST-dev` Registered Model を指しています。

{{< img src="/images/models/view_linked_model_artifacts_browser.png" alt="モデル Artifact ブラウザ" >}}  
  {{% /tab %}}
{{< /tabpane >}}