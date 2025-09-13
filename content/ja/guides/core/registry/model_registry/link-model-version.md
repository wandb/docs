---
title: Model Version をリンクする
description: モデルバージョンを W&B App で、または Python SDK を使ってプログラムで Registered Model にリンクします。
menu:
  default:
    identifier: ja-guides-core-registry-model_registry-link-model-version
    parent: model-registry
weight: 5
---

W&B App または Python SDK を使って、プログラムから モデル バージョン を registered model にリンクします。

## プログラムからモデルをリンクする

[`link_model`]({{< relref path="/ref/python/sdk/classes/run.md#link_model" lang="ja" >}}) メソッドを使用して、モデル ファイルをプログラムから W&B run にログし、[W&B Model Registry]({{< relref path="./" lang="ja" >}}) にリンクします。

`<>` で囲まれた値はご自身のものに置き換えてください。

```python
import wandb

run = wandb.init(entity="<entity>", project="<project>")
run.link_model(path="<path-to-model>", registered_model_name="<registered-model-name>")
run.finish()
```

`registered-model-name` パラメータに指定した名前の registered model がまだ存在しない場合、W&B がその registered model を作成します。

例えば、Model Registry に "Fine-Tuned-Review-Autocompletion" (`registered-model-name="Fine-Tuned-Review-Autocompletion"`) という名前の既存の registered model があるとします。そして、いくつかのモデル バージョン (`v0`、`v1`、`v2`) がそれにリンクされているとします。新しい モデル をプログラムからリンクし、同じ registered model 名 (`registered-model-name="Fine-Tuned-Review-Autocompletion"`) を使用すると、W&B はこの モデル を既存の registered model にリンクし、モデル バージョン `v3` を割り当てます。この名前の registered model が存在しない場合は、新しい registered model が作成され、モデル バージョン `v0` が割り当てられます。

["Fine-Tuned-Review-Autocompletion" registered model の例はこちら](https://wandb.ai/reviewco/registry/model?selectionPath=reviewco%2Fmodel-registry%2FFinetuned-Review-Autocompletion&view=all-models) をご覧ください。

## 対話的にモデルをリンクする
Model Registry または Artifact browser を使って、モデルを対話的にリンクします。

{{< tabpane text=true >}}
  {{% tab header="Model Registry" %}}
1. [Model Registry App](https://wandb.ai/registry/model) に移動します。
2. 新しいモデルをリンクしたい registered model の名前の横にマウスを重ねます。
3. 「**View details**」の横にあるミートボール メニュー アイコン (3 つの横向きの点) を選択します。
4. ドロップダウンから「**Link new version**」を選択します。
5. **Project** ドロップダウンから、あなたのモデルを含む project の名前を選択します。
6. **Model Artifact** ドロップダウンから、モデル アーティファクトの名前を選択します。
7. **Version** ドロップダウンから、registered model にリンクしたい モデル バージョン を選択します。

{{< img src="/images/models/link_model_wmodel_reg.gif" alt="モデル バージョンをレジストリにリンク" >}}
  {{% /tab %}}
  {{% tab header="Artifact browser" %}}
1. W&B App のあなたの project の artifact browser に移動します: `https://wandb.ai/<entity>/<project>/artifacts`
2. 左サイドバーの Artifacts アイコンを選択します。
3. Artifacts パネルから **model** ドロップダウン メニューを展開します。
4. あなたのレジストリにリンクしたい モデル バージョン をクリックします。
5. 画面の右側に表示されるモーダルから、**Select a register model** メニュードロップダウンで registered model を選択します。
6. **Next step** をクリックします。
7. (オプション) **Aliases** ドロップダウンから エイリアス を選択します。
8. **Link to registry** をクリックします。

{{< img src="/images/models/manual_linking.gif" alt="手動でモデルをリンク" >}}
  {{% /tab %}}
{{< /tabpane >}}

## リンクされたモデルのソースを表示する

リンクされた モデル のソースを表示する方法は 2 つあります: モデルがログされている project 内の artifact browser と W&B Model Registry です。

モデルレジストリ内の特定の モデル バージョン は、ソース モデル アーティファクト (モデルがログされている project 内にあります) にポインタで接続されています。ソース モデル アーティファクトにも、モデルレジストリへのポインタがあります。

{{< tabpane text=true >}}
  {{% tab header="Model Registry" %}}
1. あなたの [Model Registry App](https://wandb.ai/registry/model) に移動します。
{{< img src="/images/models/create_registered_model_1.png" alt="registered model の作成" >}}
2. あなたの registered model の名前の横にある「**View details**」を選択します。
3. **Versions** セクション内で、調査したい モデル バージョン の横にある「**View**」を選択します。
4. 右側のパネル内にある「**Version**」タブをクリックします。
5. **Version overview** セクション内に、**Source Version** フィールドを含む行があります。**Source Version** フィールドには、モデルの名前とモデルのバージョンの両方が表示されます。

例えば、次の画像は、`v0` モデル バージョンである `mnist_model` (「**Source Version**」フィールド `mnist_model:v0` を参照) が `MNIST-dev` という名前の registered model にリンクされていることを示しています。

{{< img src="/images/models/view_linked_model_registry.png" alt="レジストリ内のリンクされたモデル" >}}
  {{% /tab %}}
  {{% tab header="Artifact browser" %}}
1. W&B App のあなたの project の artifact browser に移動します: `https://wandb.ai/<entity>/<project>/artifacts`
2. 左サイドバーの Artifacts アイコンを選択します。
3. Artifacts パネルから **model** ドロップダウン メニューを展開します。
4. モデルレジストリにリンクされている モデル の名前と バージョン を選択します。
5. 右側のパネル内にある「**Version**」タブをクリックします。
6. **Version overview** セクション内に、**Linked To** フィールドを含む行があります。**Linked To** フィールドには、registered model の名前とそれが持つ バージョン (`registered-model-name:version`) の両方が表示されます。

例えば、次の画像では、`MNIST-dev` と呼ばれる registered model があります (「**Linked To**」フィールドを参照)。`v0` (`mnist_model:v0`) バージョンを持つ `mnist_model` と呼ばれる モデル バージョン が、`MNIST-dev` registered model を指しています。

{{< img src="/images/models/view_linked_model_artifacts_browser.png" alt="モデル アーティファクト ブラウザ" >}}
  {{% /tab %}}
{{< /tabpane >}}