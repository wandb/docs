---
title: Link a model version
description: W&B App を使用するか、Python SDK でプログラム的に **モデル** の **バージョン** を Registered Model
  にリンクします。
menu:
  default:
    identifier: ja-guides-core-registry-model_registry-link-model-version
    parent: model-registry
weight: 5
---

[W&B App]({{< relref path="./" lang="ja" >}})または Python SDK で、モデルのバージョンを登録済みモデルにリンクします。

## プログラムでモデルをリンクする

[`link_model`]({{< relref path="/ref/python/run.md#link_model" lang="ja" >}}) メソッドを使用して、モデルファイルをプログラムで W&B の run に ログ し、それを[W&B モデルレジストリ]({{< relref path="./" lang="ja" >}})にリンクします。

必ず `<>` で囲まれた値を独自の値に置き換えてください。

```python
import wandb

run = wandb.init(entity="<entity>", project="<project>")
run.link_model(path="<path-to-model>", registered_model_name="<登録済みモデル名>")
run.finish()
```

`registered-model-name` パラメータに指定した名前がまだ存在しない場合、W&B によって登録済みモデルが作成されます。

たとえば、モデルレジストリに「Fine-Tuned-Review-Autocompletion 」(`registered-model-name="Fine-Tuned-Review-Autocompletion"`) という名前の既存の登録済みモデルがあるとします。また、いくつかのモデルバージョンがそれ(`v0`、`v1`、`v2`)にリンクされているとします。新しいモデルをプログラムでリンクし、同じ登録済みモデル名 (`registered-model-name="Fine-Tuned-Review-Autocompletion"`) を使用すると、W&B はこのモデルを既存の登録済みモデルにリンクし、モデルバージョン `v3` を割り当てます。この名前の登録済みモデルが存在しない場合は、新しい登録済みモデルが作成され、モデルバージョン `v0` が割り当てられます。

["Fine-Tuned-Review-Autocompletion" 登録済みモデルの例はこちら](https://wandb.ai/reviewco/registry/model?selectionPath=reviewco%2Fmodel-registry%2FFinetuned-Review-Autocompletion&view=all-models)をご覧ください。

## インタラクティブにモデルをリンクする
[モデルレジストリ]({{< relref path="./" lang="ja" >}})または Artifact ブラウザを使用して、インタラクティブにモデルをリンクします。

{{< tabpane text=true >}}
  {{% tab header="モデルレジストリ" %}}
1. [https://wandb.ai/registry/model](https://wandb.ai/registry/model) のモデルレジストリ App に移動します。
2. 新しいモデルをリンクする登録済みモデルの名前の横にマウスを置きます。
3. **詳細を表示** の横にあるミートボールメニューアイコン (3 つの水平ドット) を選択します。
4. ドロップダウンから、**新しいバージョンをリンク** を選択します。
5. **Project** ドロップダウンから、モデルを含む project の名前を選択します。
6. **モデル Artifact** ドロップダウンから、モデル artifact の名前を選択します。
7. **バージョン** ドロップダウンから、登録済みモデルにリンクするモデル バージョンを選択します。

{{< img src="/images/models/link_model_wmodel_reg.gif" alt="" >}}
  {{% /tab %}}
  {{% tab header="Artifact ブラウザ" %}}
1. W&B App の project の Artifact ブラウザに移動します: `https://wandb.ai/<entity>/<project>/artifacts`
2. 左側のサイドバーにある Artifacts アイコンを選択します。
3. レジストリにリンクするモデル バージョンをクリックします。
4. **バージョンの概要** セクション内で、**レジストリにリンク** ボタンをクリックします。
5. 画面の右側に表示されるモーダルから、**登録済みモデルを選択** メニューのドロップダウンから登録済みモデルを選択します。
6. **次のステップ** をクリックします。
7. (オプション) **エイリアス** ドロップダウンからエイリアスを選択します。
8. **レジストリにリンク** をクリックします。

{{< img src="/images/models/manual_linking.gif" alt="" >}}  
  {{% /tab %}}
{{< /tabpane >}}

## リンクされたモデルのソースを表示する

リンクされたモデルのソースを表示する方法は 2 つあります。モデルが ログ されている project 内の Artifact ブラウザと W&B モデルレジストリです。

ポインタは、モデルレジストリ内の特定のモデルバージョンをソース モデル artifact (モデルが ログ されている project 内にある) に接続します。ソース モデル artifact には、モデルレジストリへのポインタもあります。

{{< tabpane text=true >}}
  {{% tab header="モデルレジストリ" %}}
1. [https://wandb.ai/registry/model](https://wandb.ai/registry/model) でモデルレジストリに移動します。
{{< img src="/images/models/create_registered_model_1.png" alt="" >}}
2. 登録済みモデルの名前の横にある **詳細を表示** を選択します。
3. **バージョン** セクション内で、調査するモデルバージョンの横にある **表示** を選択します。
4. 右側の パネル 内の **バージョン** タブをクリックします。
5. **バージョンの概要** セクション内に、**ソースバージョン** フィールドを含む行があります。**ソースバージョン** フィールドには、モデルの名前とモデルのバージョンの両方が表示されます。

たとえば、次の図は、`MNIST-dev` という名前の登録済みモデルにリンクされた `mnist_model` という名前の `v0` モデル バージョンを示しています (**ソースバージョン** フィールド `mnist_model:v0` を参照)。

{{< img src="/images/models/view_linked_model_registry.png" alt="" >}}  
  {{% /tab %}}
  {{% tab header="Artifact ブラウザ" %}}
1. W&B App の project の Artifact ブラウザに移動します: `https://wandb.ai/<entity>/<project>/artifacts`
2. 左側のサイドバーにある Artifacts アイコンを選択します。
3. Artifacts パネルから **モデル** ドロップダウン メニューを展開します。
4. モデルレジストリにリンクされているモデルの名前とバージョンを選択します。
5. 右側の パネル 内の **バージョン** タブをクリックします。
6. **バージョンの概要** セクション内に、**リンク先** フィールドを含む行があります。**リンク先** フィールドには、登録済みモデルの名前と、それが持つバージョン (`registered-model-name:version`) の両方が表示されます。

たとえば、次の図では、`MNIST-dev` という名前の登録済みモデルがあります (**リンク先** フィールドを参照)。バージョン `v0` (`mnist_model:v0`) を持つ `mnist_model` という名前のモデル バージョンは、`MNIST-dev` 登録済みモデルを指しています。

{{< img src="/images/models/view_linked_model_artifacts_browser.png" alt="" >}}  
  {{% /tab %}}
{{< /tabpane >}}
