---
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# モデルバージョンをリンクする

W&B AppまたはPython SDKを使用してプログラムで登録済みモデルにモデルバージョンをリンクします。

## プログラムでモデルをリンクする

[`link_model`](../../ref/python/run.md#link_model)メソッドを使用して、モデルファイルをW&B runにログし、[W&B Model Registry](./intro.md)にリンクします。

以下の<>で囲まれた値を自分のものに置き換えてください：

```python
import wandb

run = wandb.init(entity="<entity>", project="<project>")
run.link_model(path="<path-to-model>", registered_model_name="<registered-model-name>")
run.finish()
```

指定した`registered-model-name`パラメータの名前が既に存在しない場合、W&Bは自動的に登録済みモデルを作成します。

例えば、"Fine-Tuned-Review-Autocompletion"という名前の登録済みモデル（`registered-model-name="Fine-Tuned-Review-Autocompletion"`）がModel Registryに既に存在しているとします。そして、いくつかのモデルバージョンがリンクされているとします：`v0`、`v1`、`v2`。新しいモデルをプログラムでリンクし、同じ登録済みモデル名（`registered-model-name="Fine-Tuned-Review-Autocompletion"`）を使用すると、W&Bはこのモデルを既存の登録済みモデルにリンクし、モデルバージョン`v3`を割り当てます。この名前の登録済みモデルが存在しない場合、新しい登録済みモデルが作成され、モデルバージョン`v0`が割り当てられます。

["Fine-Tuned-Review-Autocompletion"登録済みモデルの例はこちら](https://wandb.ai/reviewco/registry/model?selectionPath=reviewco%2Fmodel-registry%2FFinetuned-Review-Autocompletion&view=all-models).

## インタラクティブにモデルをリンクする
Model RegistryまたはArtifactブラウザでインタラクティブにモデルをリンクします。

<Tabs
  defaultValue="model_ui"
  values={[
    {label: 'Model Registry', value: 'model_ui'},
    {label: 'Artifact browser', value: 'artifacts_ui'},
  ]}>
  <TabItem value="model_ui">

1. [https://wandb.ai/registry/model](https://wandb.ai/registry/model)でModel Registry Appに移動します。
2. リンクしたい新しいモデルに隣接する登録済みモデル名の横にマウスを置きます。
3. **View details**の横にある三点メニューアイコン（三つの横点）を選択します。
4. ドロップダウンから**Link new version**を選択します。
5. **Project**ドロップダウンから、モデルを含むプロジェクトの名前を選択します。
6. **Model Artifact**ドロップダウンから、モデルのアーティファクト名を選択します。
7. **Version**ドロップダウンから登録済みモデルにリンクしたいモデルバージョンを選択します。

![](/images/models/link_model_wmodel_reg.gif)

  </TabItem>
  <TabItem value="artifacts_ui">

1. W&B Appでプロジェクトのアーティファクトブラウザに移動します：`https://wandb.ai/<entity>/<project>/artifacts`
2. 左サイドバーのArtifactsアイコンを選択します。
3. 登録済みモデルにリンクしたいモデルバージョンをクリックします。
4. **Version overview**セクション内で、**Link to registry**ボタンをクリックします。
5. 画面の右側に表示されるモーダルから、**Select a register model**メニューのドロップダウンから登録済みモデルを選択します。
6. **Next step**をクリックします。
7. （任意）**Aliases**ドロップダウンからエイリアスを選択します。
8. **Link to registry**をクリックします。

![](/images/models/manual_linking.gif)

  </TabItem>
</Tabs>

## リンクされたモデルのソースを見る

リンクされたモデルのソースを見るには、モデルがログされたプロジェクト内のアーティファクトブラウザとW&B Model Registryの2つの方法があります。

ポインターがモデルレジストリ内の特定のモデルバージョンをソースモデルアーティファクトに接続します（モデルがログされたプロジェクト内にあります）。ソースモデルアーティファクトにもモデルレジストリへのポインターがあります。

<Tabs
  defaultValue="registry"
  values={[
    {label: 'Model Registry', value: 'registry'},
    {label: 'Artifact browser', value: 'browser'},
  ]}>
  <TabItem value="registry">

1. [https://wandb.ai/registry/model](https://wandb.ai/registry/model)でモデルレジストリに移動します。
![](/images/models/create_registered_model_1.png)
2. 登録済みモデルの名前の横にある**View details**を選択します。
3. **Versions**セクション内で、調査したいモデルバージョンの横にある**View**を選択します。
4. 右側のパネル内の**Version**タブをクリックします。
5. **Version overview**セクション内にある行に**Source Version**フィールドがあります。**Source Version**フィールドには、モデルの名前とバージョンが表示されます。

例えば、次の画像は、`mnist_model`という名前の`v0`モデルバージョン（**Source version**フィールド`mnist_model:v0`）が`MNIST-dev`という名前の登録済みモデルにリンクされていることを示しています。

![](/images/models/view_linked_model_registry.png)

  </TabItem>
  <TabItem value="browser">

1. W&B Appでプロジェクトのアーティファクトブラウザに移動します：`https://wandb.ai/<entity>/<project>/artifacts`
2. 左サイドバーのArtifactsアイコンを選択します。
3. Artifactsパネルから**model**ドロップダウンメニューを展開します。
4. モデルレジストリにリンクされたモデルの名前とバージョンを選択します。
5. 右側のパネル内の**Version**タブをクリックします。
6. **Version overview**セクション内にある行に**Linked To**フィールドがあります。**Linked To**フィールドには、登録済みモデルの名前とそのバージョンが表示されます（`registered-model-name:version`）。

例えば、次の画像では、`MNIST-dev`という名前の登録済みモデルが表示されています（**Linked To**フィールドを参照）。`mnist_model`という名前のモデルバージョンとバージョン`v0`（`mnist_model:v0`）が、`MNIST-dev`登録済みモデルを指しています。

![](/images/models/view_linked_model_artifacts_browser.png)

  </TabItem>
</Tabs>
