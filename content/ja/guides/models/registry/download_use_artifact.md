---
title: Download an artifact from a registry
menu:
  default:
    identifier: ja-guides-models-registry-download_use_artifact
    parent: registry
weight: 6
---

W&B Python SDK を使用して、レジストリにリンクされた artifact をダウンロードします。artifact をダウンロードして使用するには、レジストリの名前、コレクションの名前、およびダウンロードする artifact バージョンのエイリアスまたはインデックスを知る必要があります。

artifact のプロパティがわかれば、[リンクされた artifact へのパスを構築]({{< relref path="#construct-path-to-linked-artifact" lang="ja" >}}) して artifact をダウンロードできます。または、W&B App UI から [事前生成されたコードスニペットをコピーして貼り付け]({{< relref path="#copy-and-paste-pre-generated-code-snippet" lang="ja" >}}) て、レジストリにリンクされた artifact をダウンロードすることもできます。

## リンクされた artifact へのパスを構築

レジストリにリンクされた artifact をダウンロードするには、そのリンクされた artifact のパスを知っておく必要があります。パスは、レジストリ名、コレクション名、およびアクセスする artifact バージョンのエイリアスまたはインデックスで構成されます。

レジストリ、コレクション、および artifact バージョンのエイリアスまたはインデックスを取得したら、次の文字列テンプレートを使用して、リンクされた artifact へのパスを構築できます。

```python
# バージョンインデックスが指定された Artifact 名
f"wandb-registry-{REGISTRY}/{COLLECTION}:v{INDEX}"

# エイリアスが指定された Artifact 名
f"wandb-registry-{REGISTRY}/{COLLECTION}:{ALIAS}"
```

中かっこ `{}` 内の値を、アクセスするレジストリの名前、コレクション、および artifact バージョンのエイリアスまたはインデックスに置き換えます。

{{% alert %}}
artifact バージョンをコア Model Registry またはコア Dataset Registry にそれぞれリンクするには、`model` または `dataset` を指定します。
{{% /alert %}}

リンクされた artifact のパスを取得したら、`wandb.init.use_artifact` メソッドを使用して artifact にアクセスし、そのコンテンツをダウンロードします。次のコードスニペットは、W&B Registry にリンクされた artifact を使用およびダウンロードする方法を示しています。`<>` 内の値を独自の値に置き換えてください。

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
# artifact_name = '<artifact_name>' # Registry App で指定されたフルネームをコピーして貼り付け
fetched_artifact = run.use_artifact(artifact_or_name = artifact_name)  
download_path = fetched_artifact.download()  
```

`.use_artifact()` メソッドは、[run]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) を作成し、ダウンロードした artifact をその run への入力としてマークします。artifact を run への入力としてマークすると、W&B はその artifact のリネージを追跡できます。

run を作成したくない場合は、`wandb.Api()` オブジェクトを使用して artifact にアクセスできます。

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
<summary>例: W&B Registry にリンクされた artifact を使用およびダウンロードする</summary>

次のコード例は、**Fine-tuned Models** レジストリの `phi3-finetuned` というコレクションにリンクされた artifact をユーザーがダウンロードする方法を示しています。artifact バージョンのエイリアスは `production` に設定されています。

```python
import wandb

TEAM_ENTITY = "product-team-applications"
PROJECT_NAME = "user-stories"

REGISTRY = "Fine-tuned Models"
COLLECTION = "phi3-finetuned"
ALIAS = 'production'

# 指定された Team および Project 内で run を初期化
run = wandb.init(entity=TEAM_ENTITY, project = PROJECT_NAME)

artifact_name = f"wandb-registry-{REGISTRY}/{COLLECTION}:{ALIAS}"

# Artifact にアクセスし、リネージ追跡のために run への入力としてマークします
fetched_artifact = run.use_artifact(artifact_or_name = name)  

# Artifact をダウンロードします。ダウンロードしたコンテンツへのパスを返します
downloaded_path = fetched_artifact.download()  
```
</details>

可能なパラメータと戻り値の型について詳しくは、API Reference ガイドの [`use_artifact`]({{< relref path="/ref/python/run.md#use_artifact" lang="ja" >}}) と [`Artifact.download()`]({{< relref path="/ref/python/artifact#download" lang="ja" >}}) を参照してください。

{{% alert title="複数の組織に属する個人 Entity を持つ Users" %}}
複数の組織に属する個人 Entity を持つ Users は、レジストリにリンクされた Artifacts にアクセスするときに、組織の名前または Team Entity のいずれかを指定する必要があります。

```python
import wandb

REGISTRY = "<registry_name>"
COLLECTION = "<collection_name>"
VERSION = "<version>"

# Team Entity を使用して API をインスタンス化していることを確認します
api = wandb.Api(overrides={"entity": "<team-entity>"})
artifact_name = f"wandb-registry-{REGISTRY}/{COLLECTION}:{VERSION}"
artifact = api.artifact(name = artifact_name)

# パスで組織の表示名または組織 Entity を使用します
api = wandb.Api()
artifact_name = f"{ORG_NAME}/wandb-registry-{REGISTRY}/{COLLECTION}:{VERSION}"
artifact = api.artifact(name = artifact_name)
```

ここで、`ORG_NAME` は組織の表示名です。マルチテナント SaaS ユーザーは、`https://wandb.ai/account-settings/` の組織の Settings ページで組織の名前を確認できます。Dedicated Cloud および Self-Managed ユーザーは、アカウント管理者に連絡して、組織の表示名を確認してください。
{{% /alert %}}

## 事前生成されたコードスニペットをコピーして貼り付ける

W&B は、Python スクリプト、ノートブック、またはターミナルにコピーして貼り付けて、レジストリにリンクされた artifact をダウンロードできるコードスニペットを作成します。

1. Registry App に移動します。
2. artifact を含むレジストリの名前を選択します。
3. コレクションの名前を選択します。
4. artifact バージョンのリストから、アクセスするバージョンを選択します。
5. **Usage** タブを選択します。
6. **Usage API** セクションに表示されるコードスニペットをコピーします。
7. コードスニペットを Python スクリプト、ノートブック、またはターミナルに貼り付けます。

{{< img src="/images/registry/find_usage_in_registry_ui.gif" >}}
