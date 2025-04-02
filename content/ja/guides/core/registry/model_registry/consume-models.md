---
title: Download a model version
description: W&B Python SDK でモデルをダウンロードする方法
menu:
  default:
    identifier: ja-guides-core-registry-model_registry-consume-models
    parent: model-registry
weight: 8
---

W&B Python SDK を使用して、Model Registry にリンクしたモデル artifact をダウンロードします。

{{% alert %}}
モデルを再構築し、デシリアライズして、操作できる形式にするための追加の Python 関数、API 呼び出しはお客様の責任で提供する必要があります。

W&B では、モデルをメモリにロードする方法に関する情報をモデルカードに記載することをお勧めします。詳細については、[機械学習モデルのドキュメント]({{< relref path="./create-model-cards.md" lang="ja" >}}) ページを参照してください。
{{% /alert %}}

`<>` 内の values はお客様ご自身のものに置き換えてください。

```python
import wandb

# run を初期化する
run = wandb.init(project="<project>", entity="<entity>")

# モデルにアクセスしてダウンロードします。ダウンロードした artifact への path を返します
downloaded_model_path = run.use_model(name="<your-model-name>")
```

以下のいずれかの形式でモデル version を参照してください。

* `latest` - 最新の `latest` エイリアスを使用して、最も新しくリンクされたモデル version を指定します。
* `v#` - `v0`、`v1`、`v2` などを使用して、Registered Model 内の特定の version を取得します
* `alias` - ユーザー と Teams がモデル version に割り当てたカスタム alias を指定します

可能なパラメータと戻り値の型について詳しくは、API Reference ガイドの[`use_model`]({{< relref path="/ref/python/run.md#use_model" lang="ja" >}}) を参照してください。

<details>
<summary>例: ログに記録されたモデルをダウンロードして使用する</summary>

たとえば、次のコード snippet では、ユーザー が `use_model` API を呼び出しました。フェッチするモデル artifact の名前を指定し、version/alias も指定しました。次に、API から返された path を `downloaded_model_path` 変数に格納しました。

```python
import wandb

entity = "luka"
project = "NLP_Experiments"
alias = "latest"  # モデル version のセマンティックなニックネームまたは識別子
model_artifact_name = "fine-tuned-model"

# run を初期化する
run = wandb.init()
# モデルにアクセスしてダウンロードします。ダウンロードした artifact への path を返します

downloaded_model_path = run.use_model(name=f"{entity/project/model_artifact_name}:{alias}")
```
</details>

{{% alert title="2024 年に予定されている W&B Model Registry の廃止" %}}
次のタブは、まもなく廃止される Model Registry を使用してモデル artifact を消費する方法を示しています。

W&B Registry を使用して、モデル artifact を追跡、整理、消費します。詳細については、[Registry docs]({{< relref path="/guides/core/registry/" lang="ja" >}})を参照してください。
{{% /alert %}}

{{< tabpane text=true >}}
  {{% tab header="CLI" %}}
`<>` 内の values はお客様ご自身のものに置き換えてください。
```python
import wandb
# run を初期化する
run = wandb.init(project="<project>", entity="<entity>")
# モデルにアクセスしてダウンロードします。ダウンロードした artifact への path を返します
downloaded_model_path = run.use_model(name="<your-model-name>")
```
以下のいずれかの形式でモデル version を参照してください。

* `latest` - 最新の `latest` エイリアスを使用して、最も新しくリンクされたモデル version を指定します。
* `v#` - `v0`、`v1`、`v2` などを使用して、Registered Model 内の特定の version を取得します
* `alias` - ユーザー と Teams がモデル version に割り当てたカスタム alias を指定します

可能なパラメータと戻り値の型について詳しくは、API Reference ガイドの[`use_model`]({{< relref path="/ref/python/run.md#use_model" lang="ja" >}}) を参照してください。
  {{% /tab %}}
  {{% tab header="W&B App" %}}
1. [https://wandb.ai/registry/model](https://wandb.ai/registry/model) にある Model Registry App に移動します。
2. ダウンロードするモデルが含まれている登録済みモデルの名前の横にある**詳細を表示**を選択します。
3. [Versions] セクションで、ダウンロードするモデル version の横にある [View] ボタンを選択します。
4. **ファイル** タブを選択します。
5. ダウンロードするモデルファイルの横にあるダウンロードボタンをクリックします。
{{< img src="/images/models/download_model_ui.gif" alt="" >}}
  {{% /tab %}}
{{< /tabpane >}}
