---
title: モデル バージョンをダウンロードする
description: W&B Python SDK でモデルをダウンロードする方法
menu:
  default:
    identifier: consume-models
    parent: model-registry
weight: 8
---

W&B Python SDK を使って、Model Registry にリンクしたモデルのアーティファクトをダウンロードできます。

{{% alert %}}
モデルを再構築・デシリアライズして作業できる状態にするためには、追加の Python 関数や API コールを用意する必要があります。

W&B では、モデルカードを使ってモデルをメモリにロードする方法の情報をドキュメント化することを推奨しています。詳しくは [機械学習モデルのドキュメント化]({{< relref "./create-model-cards.md" >}}) ページをご覧ください。
{{% /alert %}}

`<>` 内の値はご自身のものに置き換えてください。

```python
import wandb

# run を初期化
run = wandb.init(project="<project>", entity="<entity>")

# モデルにアクセスしてダウンロードする。ダウンロードしたアーティファクトのパスを返す
downloaded_model_path = run.use_model(name="<your-model-name>")
```

モデルバージョンは、以下のいずれかの形式で指定できます。

* `latest` - 最新のバージョンを示す `latest` エイリアスを利用します。
* `v#` - Registered Model で特定のバージョンを取得するには `v0`, `v1`, `v2` などを利用します。
* `alias` - チームで割り当てたカスタムエイリアスをモデルバージョンに指定できます。

利用可能なパラメータや戻り値の詳細については、API Reference ガイドの [`use_model`]({{< relref "/ref/python/sdk/classes/run.md#use_model" >}}) を参照してください。

<details>
<summary>例：ログ済みモデルのダウンロードと利用</summary>

次のコードスニペットでは、ユーザーが `use_model` API を呼び出しています。取得したいモデルアーティファクト名と、バージョンやエイリアスも指定しています。API から返されたパスを `downloaded_model_path` 変数に格納しています。

```python
import wandb

entity = "luka"
project = "NLP_Experiments"
alias = "latest"  # モデルバージョンの意味的ニックネームまたは識別子
model_artifact_name = "fine-tuned-model"

# run を初期化
run = wandb.init()
# モデルにアクセスしてダウンロードする。ダウンロードしたアーティファクトのパスを返す

downloaded_model_path = run.use_model(name=f"{entity/project/model_artifact_name}:{alias}")
```
</details>

{{% alert title="2024年に予定されている W&B Model Registry の非推奨について" %}}
以下のタブでは、まもなく非推奨となる Model Registry でモデルアーティファクトを利用する方法を紹介します。

モデルアーティファクトを追跡・整理・利用するには、W&B Registry のご利用をおすすめします。詳細は [Registry ドキュメント]({{< relref "/guides/core/registry/" >}}) をご確認ください。
{{% /alert %}}

{{< tabpane text=true >}}
  {{% tab header="CLI" %}}
`<>` 内の値はご自身のものに置き換えてください。
```python
import wandb
# run を初期化
run = wandb.init(project="<project>", entity="<entity>")
# モデルにアクセスしてダウンロードする。ダウンロードしたアーティファクトのパスを返す
downloaded_model_path = run.use_model(name="<your-model-name>")
```
モデルバージョンは、以下のいずれかの形式で指定できます。

* `latest` - 最新のバージョンを示す `latest` エイリアスを利用します。
* `v#` - Registered Model で特定のバージョンを取得するには `v0`, `v1`, `v2` などを利用します。
* `alias` - チームで割り当てたカスタムエイリアスをモデルバージョンに指定できます。

パラメータや戻り値については API Reference ガイドの [`use_model`]({{< relref "/ref/python/sdk/classes/run.md#use_model" >}}) をご確認ください。
  {{% /tab %}}
  {{% tab header="W&B App" %}}
1. [Model Registry App](https://wandb.ai/registry/model) にアクセスします。
2. ダウンロードしたいモデルが含まれている登録済みモデルの横にある **View details** を選択します。
3. Versions セクション内で、ダウンロードしたいモデルバージョンの横にある View ボタンを選択します。
4. **Files** タブを選択します。
5. ダウンロードしたいモデルファイルの横にあるダウンロードボタンをクリックします。
{{< img src="/images/models/download_model_ui.gif" alt="UI からモデルをダウンロード" >}}
  {{% /tab %}}
{{< /tabpane >}}