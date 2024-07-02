---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# モデルバージョンをリンクする

W&BアプリまたはPython SDKでプログラム的にモデルバージョンを登録済みモデルにリンクします。

## プログラム的にモデルをリンクする

[`link_model`](../../ref/python/run.md#link_model) メソッドを使用して、プログラム的にモデルファイルをW&Bのrunにログし、 [W&B Model Registry](./intro.md) にリンクします。

`<>`で囲まれた他の値を自分のものに置き換えてください:

```python
import wandb

run = wandb.init(entity="<entity>", project="<project>")
run.link_model(path="<path-to-model>", registered_model_name="<registered-model-name>")
run.finish()
```

W&Bは、指定した `registered-model-name` パラメータの名前が既に存在しない場合、自動的に登録済みモデルを作成します。

例えば、「Fine-Tuned-Review-Autocompletion」という名前の既存の登録済みモデル (`registered-model-name="Fine-Tuned-Review-Autocompletion"`) が Model Registry にあり、いくつかのモデルバージョン `v0`, `v1`, `v2` がリンクされているとします。この状態で新しいモデルをプログラム的にリンクし、同じ登録済みモデル名 (`registered-model-name="Fine-Tuned-Review-Autocompletion"`) を使用すると、そのモデルは既存の登録済みモデルとリンクされ、 `v3` というモデルバージョンが割り当てられます。この名前の登録済みモデルが存在しない場合、新しい登録済みモデルが作成され、 `v0` というモデルバージョンが割り当てられます。

["Fine-Tuned-Review-Autocompletion" 登録済みモデルの例はこちら](https://wandb.ai/reviewco/registry/model?selectionPath=reviewco%2Fmodel-registry%2FFinetuned-Review-Autocompletion&view=all-models) をご覧ください。

## インタラクティブにモデルをリンクする
Model Registry または Artifact ブラウザを使用して、インタラクティブにモデルをリンクします。

<Tabs
  defaultValue="model_ui"
  values={[
    {label: 'Model Registry', value: 'model_ui'},
    {label: 'Artifact browser', value: 'artifacts_ui'},
  ]}>
  <TabItem value="model_ui">

1. [Model Registry App](https://wandb.ai/registry/model) にアクセスします。
2. リンクしたい新しいモデルの名前の横にマウスをホバーします。
3. 「View details」の横にあるミートボールメニューアイコン（三点リーダー）を選択します。
4. ドロップダウンメニューから「Link new version」を選択します。
5. 「Project」のドロップダウンから、モデルを含むプロジェクトの名前を選択します。
6. 「Model Artifact」のドロップダウンから、モデルアーティファクトの名前を選択します。
7. 「Version」のドロップダウンから、登録済みモデルにリンクしたいモデルバージョンを選択します。

![](/images/models/link_model_wmodel_reg.gif)

  </TabItem>
  <TabItem value="artifacts_ui">

1. W&BアプリでプロジェクトのArtifactブラウザにアクセスします: `https://wandb.ai/<entity>/<project>/artifacts`
2. 左のサイドバーでArtifactsアイコンを選択します。
3. レジストリにリンクしたいモデルバージョンをクリックします。
4. 「Version overview」セクションで、「Link to registry」ボタンをクリックします。
5. 画面右側に表示されるモーダルから、**Select a registered model** メニューのドロップダウンから登録済みモデルを選択します。
6. 「Next step」をクリックします。
7. （オプション）「Aliases」ドロップダウンからエイリアスを選択します。
8. 「Link to registry」をクリックします。

![](/images/models/manual_linking.gif)

  </TabItem>
</Tabs>

## リンク済みモデルのソースを表示

リンク済みモデルのソースを表示する方法は2つあります。モデルがログされているプロジェクト内のArtifactブラウザ、またはW&B Model Registryです。

モデルレジストリ内の特定のモデルバージョンは、そのソースモデルアーティファクト（モデルがログされているプロジェクト内に位置）にポインタで接続されています。ソースモデルアーティファクトもモデルレジストリにポインタを持っています。

<Tabs
  defaultValue="registry"
  values={[
    {label: 'Model Registry', value: 'registry'},
    {label: 'Artifact browser', value: 'browser'},
  ]}>
  <TabItem value="registry">

1. [モデルレジストリ](https://wandb.ai/registry/model) にアクセスします。
![](/images/models/create_registered_model_1.png)
2. 登録済みモデルの名前の横にある「View details」を選択します。
3. 「Versions」セクション内で、調査したいモデルバージョンの横にある「View」を選択します。
4. 右側のパネル内の「Version」タブをクリックします。
5. 「Version overview」セクション内には「Source Version」フィールドが含まれた行があります。「Source Version」フィールドには、モデルの名前とバージョンが表示されます。

例えば、以下の画像では `mnist_model:v0` というバージョンの `mnist_model` モデルが表示されており、 `MNIST-dev` という登録済みモデルにリンクされています。

![](/images/models/view_linked_model_registry.png)

  </TabItem>
  <TabItem value="browser">

1. W&BアプリでプロジェクトのArtifactブラウザにアクセスします: `https://wandb.ai/<entity>/<project>/artifacts`
2. 左のサイドバーでArtifactsアイコンを選択します。
3. Artifactsパネルの「model」ドロップダウンメニューを展開します。
4. モデルレジストリにリンクされたモデルの名前とバージョンを選択します。
5. 右側のパネル内の「Version」タブをクリックします。
6. 「Version overview」セクション内には「Linked To」フィールドが含まれた行があります。「Linked To」フィールドには、登録済みモデルの名前と、そのバージョンが表示されます（`registered-model-name:version`）。

例えば、以下の画像では `MNIST-dev` という登録済みモデルが表示されており（「Linked To」フィールド）、`mnist_model:v0` というバージョンの `mnist_model` モデルが `MNIST-dev` 登録済みモデルにポインタで接続されています。

![](/images/models/view_linked_model_artifacts_browser.png)

