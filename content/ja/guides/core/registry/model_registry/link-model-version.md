---
title: モデルバージョンをリンクする
description: W&B App または Python SDK を使って、モデル バージョンを Registered Models にリンクできます。
menu:
  default:
    identifier: ja-guides-core-registry-model_registry-link-model-version
    parent: model-registry
weight: 5
---

W&B App または Python SDK を使って、モデルバージョンを Registered Model にリンクできます。

## プログラムでモデルをリンクする

[`link_model`]({{< relref path="/ref/python/sdk/classes/run.md#link_model" lang="ja" >}}) メソッドを使うことで、モデルファイルを W&B run にログし、それを [W&B Model Registry]({{< relref path="./" lang="ja" >}}) にリンクできます。

`<>` で囲まれた値はご自身のものに置き換えてください。

```python
import wandb

run = wandb.init(entity="<entity>", project="<project>")
run.link_model(path="<path-to-model>", registered_model_name="<registered-model-name>")
run.finish()
```

`registered-model-name` パラメータで指定した名前の Registered Model がまだ存在しない場合は、W&B が自動で Registered Model を作成します。

例えば、すでに Model Registry に "Fine-Tuned-Review-Autocompletion"（`registered-model-name="Fine-Tuned-Review-Autocompletion"`）という Registered Model があるとします。ここにいくつかのモデルバージョン（`v0`、`v1`、`v2`）が紐づいている状況です。このとき、新しいモデルをプログラムでリンクし、同じ Registered Model 名（`registered-model-name="Fine-Tuned-Review-Autocompletion"`）を指定すると、W&B はこのモデルを既存の Registered Model に紐づけ、新しいモデルバージョン `v3` を割り当てます。その名前の Registered Model が存在しない場合は、新しく Registered Model が作成され、バージョンは `v0` になります。

["Fine-Tuned-Review-Autocompletion" Registered Model の例はこちら](https://wandb.ai/reviewco/registry/model?selectionPath=reviewco%2Fmodel-registry%2FFinetuned-Review-Autocompletion&view=all-models) をご覧ください。

## 対話的にモデルをリンクする

Model Registry または Artifact ブラウザを使い、対話的にモデルをリンクできます。

{{< tabpane text=true >}}
  {{% tab header="Model Registry" %}}
1. [Model Registry App](https://wandb.ai/registry/model) にアクセスします。
2. 新しくモデルをリンクしたい Registered Model の名前の横にカーソルを合わせます。
3. **View details** の横にある三点リーダー（メニュ―）アイコンをクリックします。
4. 表示されたメニューから **Link new version** を選択します。
5. **Project** ドロップダウンから、対象のモデルを含むプロジェクト名を選択します。
6. **Model Artifact** ドロップダウンから、対象のモデルアーティファクト名を選択します。
7. **Version** ドロップダウンから、Registered Model にリンクしたいモデルバージョンを選択します。

{{< img src="/images/models/link_model_wmodel_reg.gif" alt="モデルバージョンをレジストリにリンクする" >}}
  {{% /tab %}}
  {{% tab header="Artifact browser" %}}
1. W&B App で、プロジェクトのアーティファクトブラウザにアクセスします： `https://wandb.ai/<entity>/<project>/artifacts`
2. 左サイドバーから Artifacts アイコンをクリックします。
3. レジストリにリンクしたいモデルバージョンをクリックします。
4. **Version overview** セクション内の **Link to registry** ボタンをクリックします。
5. 画面右に表示されるモーダル内の **Select a register model** メニューから Registered Model を選びます。
6. **Next step** をクリックします。
7. （オプション）**Aliases** ドロップダウンからエイリアスを選択します。
8. **Link to registry** をクリックします。

{{< img src="/images/models/manual_linking.gif" alt="手動でモデルリンク" >}}  
  {{% /tab %}}
{{< /tabpane >}}



## リンク済みモデルのソースを見る

リンク済みモデルのソースを見る方法は2つあります：モデルがログされているプロジェクト内のアーティファクトブラウザ、または W&B Model Registry です。

ポインターはモデールレジストリ内の特定のモデルバージョンと、そのソースモデルアーティファクト（モデルがログされたプロジェクト内）を結びつけます。ソースモデルアーティファクトにも、モデルレジストリへのポインターが付きます。

{{< tabpane text=true >}}
  {{% tab header="Model Registry" %}}
1. [Model Registry App](https://wandb.ai/registry/model) にアクセスします。
{{< img src="/images/models/create_registered_model_1.png" alt="Registered Model の作成" >}}
2. Registered Model 名の横にある **View details** をクリックします。
3. **Versions** セクションから、調べたいモデルバージョンの右側にある **View** をクリックします。
4. 右側パネルの **Version** タブを選択します。
5. **Version overview** セクションにある **Source Version** 行で、モデル名とバージョンが確認できます。

例えば、下記画像のように `mnist_model` という名前の `v0` モデルバージョン（**Source version** フィールド `mnist_model:v0`）が、`MNIST-dev` という Registered Model にリンクされています。

{{< img src="/images/models/view_linked_model_registry.png" alt="レジストリ内のリンク済みモデル" >}}  
  {{% /tab %}}
  {{% tab header="Artifact browser" %}}
1. W&B App で、プロジェクトのアーティファクトブラウザにアクセスします： `https://wandb.ai/<entity>/<project>/artifacts`
2. 左サイドバーより Artifacts アイコンをクリックします。
3. Artifacts パネルの **model** ドロップダウンを展開します。
4. モデルレジストリにリンクされているモデルの名前とバージョンを選択します。
5. 右パネル内の **Version** タブをクリックします。
6. **Version overview** セクション内、**Linked To** 行で Registered Model 名とそのバージョン（`registered-model-name:version`）を確認できます。

例えば、以下の画像では `MNIST-dev` という Registered Model（**Linked To** フィールド参照）と、`v0` バージョンの `mnist_model`（`mnist_model:v0`）が紐付いていることがわかります。

{{< img src="/images/models/view_linked_model_artifacts_browser.png" alt="モデルアーティファクトブラウザ" >}}  
  {{% /tab %}}
{{< /tabpane >}}