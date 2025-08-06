---
title: レジストリからアーティファクトをダウンロードする
menu:
  default:
    identifier: ja-guides-core-registry-download_use_artifact
    parent: registry
weight: 6
---

W&B Python SDK を使って、レジストリにリンクされた artifact をダウンロードできます。artifact をダウンロード・利用するには、レジストリ名、コレクション名、そしてダウンロードしたい artifact バージョンのエイリアスまたはインデックスを知っておく必要があります。

artifact の情報が分かったら、[リンクされた artifact のパスを構築]({{< relref path="#construct-path-to-linked-artifact" lang="ja" >}})し、artifact をダウンロードできます。または、W&B App UI から [あらかじめ生成されたコードスニペットをコピー＆ペースト]({{< relref path="#copy-and-paste-pre-generated-code-snippet" lang="ja" >}})して、レジストリにリンクされた artifact をダウンロードすることもできます。

## リンクされた artifact のパスを構築する

レジストリにリンクされた artifact をダウンロードするには、その artifact のパスを知っている必要があります。パスは、レジストリ名・コレクション名・そしてアクセスしたい artifact バージョンのエイリアスまたはインデックスから構成されます。

レジストリ、コレクション、artifact バージョンのエイリアスまたはインデックスが分かれば、以下の文字列テンプレートを使ってリンクされた artifact のパスを作成できます。

```python
# バージョンインデックスを指定した場合の artifact 名
f"wandb-registry-{REGISTRY}/{COLLECTION}:v{INDEX}"

# エイリアスを指定した場合の artifact 名
f"wandb-registry-{REGISTRY}/{COLLECTION}:{ALIAS}"
```

波括弧 `{}` の中身を、自分がアクセスしたいレジストリ名、コレクション名、および artifact バージョンのエイリアスまたはインデックスに置き換えてください。

{{% alert %}}
artifact バージョンをコアの Model registry・Dataset registry に紐付けるには、それぞれ `model` または `dataset` を指定してください。
{{% /alert %}}

リンクされた artifact のパスが分かったら、`wandb.init.use_artifact` メソッドを使って artifact にアクセスし、その内容をダウンロードできます。下記のコードスニペットで、値の部分（ `<>` ）は自分のものに置き換えて利用してください。

```python
import wandb

REGISTRY = '<registry_name>'
COLLECTION = '<collection_name>'
ALIAS = '<artifact_alias>'

run = wandb.init(
   entity = '<team_name>',
   project = '<project_name>'
   )  

artifact_name = f"wandb-registry-{REGISTRY}/{COLLECTION}:{ALIAS}"
# artifact_name = '<artifact_name>' # Registry App 上に表示されるフルネームをそのままコピペしても OK
fetched_artifact = run.use_artifact(artifact_or_name = artifact_name)  
download_path = fetched_artifact.download()  
```

`.use_artifact()` メソッドは [run]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) を作成し、ダウンロードした artifact をその run の入力としてマークします。artifact を run の入力としてマークすることで、W&B が artifact のリネージを追跡できるようになります。

もし run を作成したくない場合は、`wandb.Api()` オブジェクトを使って artifact にアクセスできます。

```python
import wandb

REGISTRY = "<registry_name>"
COLLECTION = "<collection_name>"
VERSION = "<version>"

api = wandb.Api()
artifact_name = f"wandb-registry-{REGISTRY}/{COLLECTION}:{VERSION}"
artifact = api.artifact(name = artifact_name)
```

<details>
<summary>例：W&B Registry にリンクされた artifact を利用・ダウンロードする</summary>

以下のコード例は、**Fine-tuned Models** レジストリ内の `phi3-finetuned` というコレクションにリンクされた artifact をダウンロードするものです。artifact バージョンのエイリアスは `production` です。

```python
import wandb

TEAM_ENTITY = "product-team-applications"
PROJECT_NAME = "user-stories"

REGISTRY = "Fine-tuned Models"
COLLECTION = "phi3-finetuned"
ALIAS = 'production'

# 指定した team・project 内で run を初期化
run = wandb.init(entity=TEAM_ENTITY, project = PROJECT_NAME)

artifact_name = f"wandb-registry-{REGISTRY}/{COLLECTION}:{ALIAS}"

# artifact にアクセスし、run の入力としてリネージ追跡
fetched_artifact = run.use_artifact(artifact_or_name = name)  

# artifact をダウンロード。ダウンロード先のパスを返す
downloaded_path = fetched_artifact.download()  
```
</details>

API リファレンス内の [`use_artifact`]({{< relref path="/ref/python/sdk/classes/run.md#use_artifact" lang="ja" >}}) および [`Artifact.download()`]({{< relref path="/ref/python/sdk/classes/artifact.md#download" lang="ja" >}}) で、パラメータや返り値の詳細を確認できます。

{{% alert title="複数の組織に所属する個人 entity を持つユーザーの方へ" %}} 
複数の組織に所属する個人 entity を持つユーザーは、レジストリにリンクされた artifact にアクセスする際、組織名を指定するか、team entity を利用する必要があります。

```python
import wandb

REGISTRY = "<registry_name>"
COLLECTION = "<collection_name>"
VERSION = "<version>"

# API インスタンス化の際、自分の team entity も指定
api = wandb.Api(overrides={"entity": "<team-entity>"})
artifact_name = f"wandb-registry-{REGISTRY}/{COLLECTION}:{VERSION}"
artifact = api.artifact(name = artifact_name)

# org 表示名または entity 名をパスに使用
api = wandb.Api()
artifact_name = f"{ORG_NAME}/wandb-registry-{REGISTRY}/{COLLECTION}:{VERSION}"
artifact = api.artifact(name = artifact_name)
```

ここで `ORG_NAME` はご自身の組織の表示名（display name）です。マルチテナント SaaS ユーザーは、組織設定ページ `https://wandb.ai/account-settings/` から組織名を確認できます。専用クラウドやセルフマネージド環境の場合は、組織表示名についてアカウント管理者にご確認ください。
{{% /alert %}}

## あらかじめ生成されたコードスニペットをコピー＆ペースト

W&B では、artifact をレジストリからダウンロードするために Python スクリプト・ノートブック・ターミナルに直接貼り付けて使えるコードスニペットを自動生成しています。

1. Registry App にアクセスします。
2. 自分の artifact が含まれるレジストリ名を選択します。
3. コレクション名を選択します。
4. artifact バージョン一覧からアクセスしたいバージョンを選択します。
5. **Usage** タブをクリックします。
6. **Usage API** セクションに表示されているコードスニペットをコピーします。
7. コピーしたコードスニペットを Python スクリプト、ノートブック、ターミナルに貼り付けて利用します。

{{< img src="/images/registry/find_usage_in_registry_ui.gif" >}}