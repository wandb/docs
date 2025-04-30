---
title: モデルバージョンをダウンロードする
description: W&B Python SDK で モデル をダウンロードする方法
menu:
  default:
    identifier: ja-guides-core-registry-model_registry-consume-models
    parent: model-registry
weight: 8
---

W&B Python SDK を使用して、Model Registry にリンクしたモデルアーティファクトをダウンロードします。

{{% alert %}}
モデルを再構築し、逆シリアル化して作業可能な形式に変換するための Python 関数や API コールの提供はユーザーの責任です。

W&B はモデルカードを使って、モデルをメモリにロードする方法の情報を文書化することを推奨しています。詳細は、[Document machine learning models]({{< relref path="./create-model-cards.md" lang="ja" >}}) ページをご覧ください。
{{% /alert %}}

`<>` の中の値を自身のものに置き換えてください：

```python
import wandb

# Run を初期化
run = wandb.init(project="<project>", entity="<entity>")

# モデルへのアクセスとダウンロード。ダウンロードしたアーティファクトへのパスを返します
downloaded_model_path = run.use_model(name="<your-model-name>")
```

モデルバージョンを以下のいずれかの形式で参照します：

* `latest` - 最も最近リンクされたモデルバージョンを指定するために `latest` エイリアスを使用します。
* `v#` - 特定のバージョンを取得するために `v0`、`v1`、`v2` などを使用します。
* `alias` - モデルバージョンに対してチームが設定したカスタムエイリアスを指定します。

API リファレンスガイドの [`use_model`]({{< relref path="/ref/python/run.md#use_model" lang="ja" >}}) を参照して、使用可能なパラメータと返り値の型についての詳細を確認してください。

<details>
<summary>例：ログされたモデルをダウンロードして使用する</summary>

例えば、以下のコードスニペットでは、ユーザーが `use_model` API を呼び出しています。彼らは取得したいモデルアーティファクトの名前を指定し、さらにバージョン/エイリアスも提供しています。その後、API から返されたパスを `downloaded_model_path` 変数に格納しています。

```python
import wandb

entity = "luka"
project = "NLP_Experiments"
alias = "latest"  # モデルバージョンのセマンティックニックネームまたは識別子
model_artifact_name = "fine-tuned-model"

# Run を初期化
run = wandb.init()
# モデルへのアクセスとダウンロード。ダウンロードしたアーティファクトへのパスを返します

downloaded_model_path = run.use_model(name=f"{entity/project/model_artifact_name}:{alias}")
```
</details>

{{% alert title="2024年のW&B Model Registryの廃止予定について" %}}
以下のタブでは、近日廃止予定の Model Registry を使用してモデルアーティファクトを利用する方法を示しています。

W&B Registry を使用して、モデルアーティファクトを追跡、整理、利用します。詳細は [Registry docs]({{< relref path="/guides/core/registry/" lang="ja" >}}) を参照してください。
{{% /alert %}}

{{< tabpane text=true >}}
  {{% tab header="CLI" %}}
`<>` の中の値を自身のものに置き換えてください：
```python
import wandb
# Run を初期化
run = wandb.init(project="<project>", entity="<entity>")
# モデルへのアクセスとダウンロード。ダウンロードしたアーティファクトへのパスを返します
downloaded_model_path = run.use_model(name="<your-model-name>")
```
モデルバージョンを以下のいずれかの形式で参照します：

* `latest` - 最も最近リンクされたモデルバージョンを指定するために `latest` エイリアスを使用します。
* `v#` - 特定のバージョンを取得するために `v0`、`v1`、`v2` などを使用します。
* `alias` - モデルバージョンに対してチームが設定したカスタムエイリアスを指定します。

API リファレンスガイドの [`use_model`]({{< relref path="/ref/python/run.md#use_model" lang="ja" >}}) を参照して、使用可能なパラメータと返り値の型についての詳細を確認してください。  
  {{% /tab %}}
  {{% tab header="W&B App" %}}
1. [https://wandb.ai/registry/model](https://wandb.ai/registry/model) の Model Registry App に移動します。
2. ダウンロードしたいモデルを含む登録済みモデル名の隣にある **詳細を見る** を選択します。
3. バージョンセクション内で、ダウンロードしたいモデルバージョンの隣にある表示ボタンを選択します。
4. **ファイル** タブを選択します。
5. ダウンロードしたいモデルファイルの隣にあるダウンロードボタンをクリックします。 
{{< img src="/images/models/download_model_ui.gif" alt="" >}}  
  {{% /tab %}}
{{< /tabpane >}}