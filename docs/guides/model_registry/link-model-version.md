---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# モデルバージョンをリンクする

モデルバージョンをW&Bの登録済みモデルにリンクするには、W&BアプリやPython SDKを使ってプログラム的に行うことができます。

## プログラム的にモデルをリンクする

[`link_model`](../../ref/python/run.md#link_model) メソッドを使って、W&Bのrunにモデルファイルをプログラム的にログし、[W&B Model Registry](./intro.md)にリンクします。

`<>` で囲まれた他の値を自分のものに置き換えてください:

```python
import wandb

run = wandb.init(entity="<entity>", project="<project>")
run.link_model(path="<path-to-model>", registered_model_name="<registered-model-name>")
run.finish()
```

`registered-model-name` パラメータに指定した名前が既に存在しない場合、W&Bはあなたのために登録済みモデルを作成します。

例えば、"Fine-Tuned-Review-Autocompletion"(`registered-model-name="Fine-Tuned-Review-Autocompletion"`)という名前の登録済みモデルがあり、いくつかのモデルバージョン (`v0`, `v1`, `v2`) がリンクされているとします。新しいモデルをプログラム的にリンクし、同じ登録済みモデル名(`registered-model-name="Fine-Tuned-Review-Autocompletion"`)を使用すると、W&Bはこのモデルを既存の登録済みモデルにリンクし、モデルバージョン `v3` を割り当てます。この名前の登録済みモデルが存在しない場合、新しい登録済みモデルが作成され、モデルバージョン `v0` を持ちます。

["Fine-Tuned-Review-Autocompletion" 登録済みモデルの例はこちら](https://wandb.ai/reviewco/registry/model?selectionPath=reviewco%2Fmodel-registry%2FFinetuned-Review-Autocompletion&view=all-models)

## インタラクティブにモデルをリンクする
Model Registry もしくは Artifact ブラウザを使ってインタラクティブにモデルをリンクします。

<Tabs
  defaultValue="model_ui"
  values={[
    {label: 'Model Registry', value: 'model_ui'},
    {label: 'Artifact browser', value: 'artifacts_ui'},
  ]}>
  <TabItem value="model_ui">

1. [https://wandb.ai/registry/model](https://wandb.ai/registry/model) にある Model Registry App に移動します。
2. 登録済みモデルの名前の横にマウスをホバーします。
3.  **View details** アイコン（3点リーダー）を選択します。
4. ドロップダウンから **Link new version** を選択します。
5. **Project** ドロップダウンから、モデルを含むプロジェクトの名前を選択します。
6. **Model Artifact** ドロップダウンから、モデルアーティファクトの名前を選択します。
7. **Version** ドロップダウンから、登録済みモデルにリンクしたいモデルバージョンを選択します。

![](/images/models/link_model_wmodel_reg.gif)

  </TabItem>
  <TabItem value="artifacts_ui">

1. あなたのプロジェクトのアーティファクト ブラウザに移動します: `https://wandb.ai/<entity>/<project>/artifacts`
2. 左側のサイドバーでアーティファクトアイコンを選択します。
3. 登録にリンクしたいモデルバージョンをクリックします。
4. **Version overview** セクション内で **Link to registry** ボタンをクリックします。
5. 画面の右側に表示されるモーダルから、 **Select a register model** メニューのドロップダウンから、登録済みモデルを選択します。
6. **Next step** をクリックします。
7. （任意）**Aliases** ドロップダウンからエイリアスを選択します。
8. **Link to registry** をクリックします。

![](/images/models/manual_linking.gif)

  </TabItem>
</Tabs>

## リンクされたモデルのソースを見る

リンクされたモデルのソースを見る方法は2つあります: モデルがログされたプロジェクト内のアーティファクトブラウザとW&B Model Registryです。

ポインタはモデルレジストリ内の特定のモデルバージョンを、モデルがログされたプロジェクト内にあるソースモデルアーティファクトに接続します。ソースモデルアーティファクトにもモデルレジストリへのポインタがあります。

<Tabs
  defaultValue="registry"
  values={[
    {label: 'Model Registry', value: 'registry'},
    {label: 'Artifact browser', value: 'browser'},
  ]}>
  <TabItem value="registry">

1. あなたのモデルレジストリに移動します: [https://wandb.ai/registry/model](https://wandb.ai/registry/model)
![](/images/models/create_registered_model_1.png)
2. 登録済みモデルの名前の横にある **View details** を選択します。
3. **Versions** セクション内で、調査したいモデルバージョンの横にある **View** を選択します。
4. 右パネル内の **Version** タブをクリックします。
5. **Version overview** セクション内に **Source Version** フィールドを持つ行があります。これはモデルの名前とモデルのバージョンの両方を表示します。

例えば、下記画像では `v0` モデルバージョンで呼ばれる `mnist_model` ( **Source version** フィールド `mnist_model:v0` を参照) が、 `MNIST-dev` という登録モデルにリンクされています。

![](/images/models/view_linked_model_registry.png)

  </TabItem>
  <TabItem value="browser">

1. あなたのプロジェクトのアーティファクト ブラウザに移動します: `https://wandb.ai/<entity>/<project>/artifacts`
2. 左側のサイドバーでアーティファクトアイコンを選択します。
3. アーティファクト パネルから **model** ドロップダウンメニューを展開します。
4. モデルレジストリにリンクされた名前とバージョンのモデルを選択します。
5. 右パネル内の **Version** タブをクリックします。
6. **Version overview** セクション内に **Linked To** フィールドを持つ行があります。 これは登録済みモデルの名前とそのバージョン (`registered-model-name:version`) を表示します。

例えば、下記画像では `MNIST-dev` という登録済みモデルがあります ( **Linked To** フィールドを参照)。バージョン `v0` (`mnist_model:v0`) のモデルバージョンが `MNIST-dev` 登録済みモデルを指しています。

![](/images/models/view_linked_model_artifacts_browser.png)

  </TabItem>
</Tabs>