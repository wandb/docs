---
title: モデル バージョンをダウンロードする
description: W&B Python SDK を使ってモデルをダウンロードする方法
menu:
  default:
    identifier: ja-guides-core-registry-model_registry-consume-models
    parent: model-registry
weight: 8
---

W&B Python SDK を使って、Model Registry にリンクしたモデルアーティファクトをダウンロードできます。

{{% alert %}}
追加の Python 関数や API コールを使って、モデルを再構築・デシリアライズする作業はユーザー自身が行う必要があります。

W&B では、モデルカードを使ってモデルをメモリにロードする方法などの情報をドキュメント化することを推奨しています。詳細は[機械学習モデルのドキュメント化]({{< relref path="./create-model-cards.md" lang="ja" >}})のページをご覧ください。
{{% /alert %}}

`<>` 内の値は、ご自身の情報に置き換えてください。

```python
import wandb

# run を初期化
run = wandb.init(project="<project>", entity="<entity>")

# モデルにアクセスしてダウンロード。ダウンロードしたアーティファクトのパスが返ります
downloaded_model_path = run.use_model(name="<your-model-name>")
```

モデルバージョンを指定する際は、以下のいずれかの形式を利用できます:

* `latest` - `latest` エイリアスを使って、最新でリンクされたモデルバージョンを指定します。
* `v#` - `v0`、`v1`、`v2` など、Registered Model 内の特定のバージョンを指定します。
* `alias` - あなたやチームがそのモデルバージョンに割り当てたカスタムエイリアスを指定します。

利用可能なパラメータや返り値の型など、詳細は API Reference ガイドの [`use_model`]({{< relref path="/ref/python/sdk/classes/run.md#use_model" lang="ja" >}}) をご参照ください。

<details>
<summary>例: ログ済みのモデルをダウンロードして利用する</summary>

例えば、以下のコードスニペットでは、ユーザーが `use_model` API を呼び出しています。取得したいモデルアーティファクトの名前を指定し、さらにバージョンやエイリアスも与えています。その後、API から返されたパスを `downloaded_model_path` という変数に格納しています。

```python
import wandb

entity = "luka"
project = "NLP_Experiments"
alias = "latest"  # モデルバージョンのセマンティックなニックネームや識別子
model_artifact_name = "fine-tuned-model"

# run を初期化
run = wandb.init()
# モデルにアクセスしてダウンロード。ダウンロードしたアーティファクトのパスが返ります

downloaded_model_path = run.use_model(name=f"{entity/project/model_artifact_name}:{alias}")
```
</details>

{{% alert title="2024 年に予定されている W&B Model Registry の廃止について" %}}
以下のタブでは、まもなく廃止予定の Model Registry を使用してモデルアーティファクトを活用する方法を紹介しています。

モデルアーティファクトの追跡・整理・活用には、W&B Registry をご利用ください。詳細は [Registry docs]({{< relref path="/guides/core/registry/" lang="ja" >}}) をご参照ください。
{{% /alert %}}

{{< tabpane text=true >}}
  {{% tab header="CLI" %}}
`<>` 内の値は、ご自身の情報に置き換えてください。
```python
import wandb
# run を初期化
run = wandb.init(project="<project>", entity="<entity>")
# モデルにアクセスしてダウンロード。ダウンロードしたアーティファクトのパスが返ります
downloaded_model_path = run.use_model(name="<your-model-name>")
```
モデルバージョンを指定する際は以下のいずれかの形式を利用できます:

* `latest` - `latest` エイリアスを使って、最新でリンクされたモデルバージョンを指定します。
* `v#` - `v0`、`v1`、`v2` など、Registered Model 内の特定のバージョンを指定します。
* `alias` - あなたやチームがそのモデルバージョンに割り当てたカスタムエイリアスを指定します。

パラメータや返り値については、API Reference ガイドの [`use_model`]({{< relref path="/ref/python/sdk/classes/run.md#use_model" lang="ja" >}}) をご参照ください。
  {{% /tab %}}
  {{% tab header="W&B App" %}}
1. [Model Registry App](https://wandb.ai/registry/model) にアクセスします。
2. ダウンロードしたいモデルが含まれている registered model の横にある **View details** を選択します。
3. Versions セクションで、ダウンロードしたいモデルバージョンの横にある View ボタンをクリックします。
4. **Files** タブを選択します。
5. ダウンロードしたいモデルファイルの横にあるダウンロードボタンをクリックします。
{{< img src="/images/models/download_model_ui.gif" alt="Download model from UI" >}}  
  {{% /tab %}}
{{< /tabpane >}}