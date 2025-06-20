---
title: モデルバージョンをリンクする
description: モデル バージョンを登録されたモデルに、W&B アプリまたは Python SDK を使ってプログラム的にリンクします。
menu:
  default:
    identifier: ja-guides-core-registry-model_registry-link-model-version
    parent: model-registry
weight: 5
---

モデルのバージョンを W&B App または Python SDK を使用してプログラムで登録済みのモデルにリンクします。

## プログラムでモデルをリンクする

[`link_model`]({{< relref path="/ref/python/run.md#link_model" lang="ja" >}}) メソッドを使用して、プログラムでモデルファイルを W&B run にログし、それを [W&B モデルレジストリ]({{< relref path="./" lang="ja" >}}) にリンクします。

`<>`で囲まれた値を自分のものに置き換えることを忘れないでください:

```python
import wandb

run = wandb.init(entity="<entity>", project="<project>")
run.link_model(path="<path-to-model>", registered_model_name="<registered-model-name>")
run.finish()
```

指定した `registered-model-name` パラメータの名前が既に存在しない場合、W&B は登録済みのモデルを自動的に作成します。

例えば、既に "Fine-Tuned-Review-Autocompletion" という名前の登録済みモデル(`registered-model-name="Fine-Tuned-Review-Autocompletion"`)がモデルレジストリにあり、それにいくつかのモデルバージョンがリンクされているとします: `v0`、`v1`、`v2`。新しいモデルをプログラムでリンクし、同じ登録済みモデル名を使用した場合（`registered-model-name="Fine-Tuned-Review-Autocompletion"`）、W&B はこのモデルを既存の登録済みモデルにリンクし、モデルバージョン `v3` を割り当てます。この名前の登録済みモデルが存在しない場合、新しい登録済みモデルが作成され、モデルバージョン `v0` を持ちます。

["Fine-Tuned-Review-Autocompletion" 登録済みモデルの一例をここでご覧ください](https://wandb.ai/reviewco/registry/model?selectionPath=reviewco%2Fmodel-registry%2FFinetuned-Review-Autocompletion&view=all-models).

## インタラクティブにモデルをリンクする
インタラクティブにモデルレジストリまたはアーティファクトブラウザでモデルをリンクします。

{{< tabpane text=true >}}
  {{% tab header="Model Registry" %}}
1. [https://wandb.ai/registry/model](https://wandb.ai/registry/model) のモデルレジストリアプリに移動します。
2. 新しいモデルをリンクしたい登録済みモデルの名前の横にマウスをホバーします。
3. **View details** の横のミートボールメニューアイコン（三つの水平な点）を選択します。
4. ドロップダウンメニューから **Link new version** を選択します。
5. **Project** ドロップダウンからモデルを含むプロジェクトの名前を選択します。
6. **Model Artifact** ドロップダウンからモデルアーティファクトの名前を選択します。
7. **Version** ドロップダウンから登録済みモデルにリンクしたいモデルバージョンを選択します。

{{< img src="/images/models/link_model_wmodel_reg.gif" alt="" >}}
  {{% /tab %}}
  {{% tab header="Artifact browser" %}}
1. W&B App でプロジェクトのアーティファクトブラウザに移動します: `https://wandb.ai/<entity>/<project>/artifacts`
2. 左側のサイドバーで Artifacts アイコンを選択します。
3. リストにあなたのモデルを表示したいプロジェクトを表示します。
4. モデルのバージョンをクリックして、モデルレジストリにリンクします。
5. 画面右側に表示されるモーダルから、**Select a register model** メニュードロップダウンから登録済みモデルを選択します。
6. **Next step** をクリックします。
7. （オプション）**Aliases** ドロップダウンからエイリアスを選択します。
8. **Link to registry** をクリックします。

{{< img src="/images/models/manual_linking.gif" alt="" >}}  
  {{% /tab %}}
{{< /tabpane >}}


## リンクされたモデルのソースを表示する

リンクされたモデルのソースを表示する方法は2つあります: モデルがログされているプロジェクト内のアーティファクトブラウザと W&B モデルレジストリです。

モデルレジストリ内の特定のモデルバージョンを、（そのモデルがログされているプロジェクト内に位置する）ソースモデルアーティファクトと接続するポインタがあります。ソースモデルアーティファクトにもモデルレジストリへのポインタがあります。

{{< tabpane text=true >}}
  {{% tab header="Model Registry" %}}
1. [https://wandb.ai/registry/model](https://wandb.ai/registry/model) でモデルレジストリに移動します。
{{< img src="/images/models/create_registered_model_1.png" alt="" >}}
2. 登録済みモデルの名前の横で **View details** を選択します。
3. **Versions** セクション内で調査したいモデルバージョンの横にある **View** を選択します。
4. 右パネル内の **Version** タブをクリックします。
5. **Version overview** セクション内に **Source Version** フィールドを含む行があります。**Source Version** フィールドはモデルの名前とそのバージョンを示しています。

例えば、次の画像は `v0` モデルバージョンである `mnist_model` （**Source version** フィールド `mnist_model:v0` を参照）を登録済みモデル `MNIST-dev` にリンクしていることを示しています。

{{< img src="/images/models/view_linked_model_registry.png" alt="" >}}  
  {{% /tab %}}
  {{% tab header="Artifact browser" %}}
1. W&B App でプロジェクトのアーティファクトブラウザに移動します: `https://wandb.ai/<entity>/<project>/artifacts`
2. 左側のサイドバーで Artifacts アイコンを選択します。
3. アーティファクトパネルから **model** ドロップダウンメニューを展開します。
4. モデルレジストリにリンクされたモデルの名前とバージョンを選択します。
5. 右パネル内の **Version** タブをクリックします。
6. **Version overview** セクション内に **Linked To** フィールドを含む行があります。**Linked To** フィールドは、登録済みモデルの名前とそれに属するバージョンを示しています（`registered-model-name:version`）。

例えば、次の画像では、`MNIST-dev` という登録済みモデルがあります（**Linked To** フィールドを参照）。バージョン `v0` のモデルバージョン `mnist_model`（`mnist_model:v0`）が `MNIST-dev` 登録済みモデルを指しています。

{{< img src="/images/models/view_linked_model_artifacts_browser.png" alt="" >}}  
  {{% /tab %}}
{{< /tabpane >}}