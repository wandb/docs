---
title: レジストリからアーティファクトをダウンロードする
menu:
  default:
    identifier: ja-guides-core-registry-download_use_artifact
    parent: registry
weight: 6
---

W&B Python SDK を使用して、レジストリにリンクされたアーティファクトをダウンロードします。アーティファクトをダウンロードして使用するには、レジストリ名、コレクション名、およびダウンロードしたいアーティファクトバージョンのエイリアスまたはインデックスを知る必要があります。

アーティファクトのプロパティを知ったら、[リンクされたアーティファクトへのパスを構築]({{< relref path="#construct-path-to-linked-artifact" lang="ja" >}})してアーティファクトをダウンロードできます。または、W&B アプリ UI から事前に生成されたコードスニペットを[コピーして貼り付け]({{< relref path="#copy-and-paste-pre-generated-code-snippet" lang="ja" >}})することで、レジストリにリンクされたアーティファクトをダウンロードすることもできます。

## リンクされたアーティファクトへのパスを構築

レジストリにリンクされたアーティファクトをダウンロードするには、そのリンクされたアーティファクトのパスを知っている必要があります。パスは、レジストリ名、コレクション名、およびアクセスしたいアーティファクトバージョンのエイリアスまたはインデックスで構成されます。

レジストリ、コレクション、およびアーティファクトバージョンのエイリアスまたはインデックスを手に入れたら、以下の文字列テンプレートを使用してリンクされたアーティファクトへのパスを構築できます。

```python
# バージョンインデックスを指定したアーティファクト名
f"wandb-registry-{REGISTRY}/{COLLECTION}:v{INDEX}"

# エイリアスを指定したアーティファクト名
f"wandb-registry-{REGISTRY}/{COLLECTION}:{ALIAS}"
```

中括弧 `{}` 内の値を、アクセスしたいレジストリ、コレクション、およびアーティファクトバージョンのエイリアスまたはインデックスの名前で置き換えてください。

{{% alert %}}
アーティファクトバージョンをコアモデルレジストリまたはコアデータセットレジストリにリンクするには、`model` または `dataset` を指定してください。
{{% /alert %}}

リンクされたアーティファクトのパスを取得したら、`wandb.init.use_artifact` メソッドを使用してアーティファクトにアクセスし、その内容をダウンロードします。以下のコードスニペットは、W&B レジストリにリンクされたアーティファクトを使用およびダウンロードする方法を示しています。`<>` 内の値を自分のものに置き換えてください。

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

`.use_artifact()` メソッドは、[run]({{< relref path="/guides/models/track/runs/" lang="ja" >}})を作成するとともに、ダウンロードしたアーティファクトをその run の入力としてマークします。 アーティファクトを run の入力としてマークすることにより、W&B はそのアーティファクトのリネージを追跡できます。

runを作成したくない場合は、`wandb.Api()` オブジェクトを使用してアーティファクトにアクセスできます。

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
<summary>例: W&B レジストリにリンクされたアーティファクトを使用およびダウンロード</summary>

次のコード例は、ユーザーが **Fine-tuned Models** レジストリにある `phi3-finetuned` というコレクションにリンクされたアーティファクトをダウンロードする方法を示しています。アーティファクトバージョンのエイリアスは `production` に設定されています。

```python
import wandb

TEAM_ENTITY = "product-team-applications"
PROJECT_NAME = "user-stories"

REGISTRY = "Fine-tuned Models"
COLLECTION = "phi3-finetuned"
ALIAS = 'production'

# 指定されたチームとプロジェクト内で run を初期化
run = wandb.init(entity=TEAM_ENTITY, project = PROJECT_NAME)

artifact_name = f"wandb-registry-{REGISTRY}/{COLLECTION}:{ALIAS}"

# アーティファクトにアクセスし、それをリネージ追跡のために run の入力としてマーク
fetched_artifact = run.use_artifact(artifact_or_name = name)  

# アーティファクトをダウンロード。ダウンロードされたコンテンツのパスを返します
downloaded_path = fetched_artifact.download()  
```
</details>

APIリファレンスガイドの [`use_artifact`]({{< relref path="/ref/python/run.md#use_artifact" lang="ja" >}}) と [`Artifact.download()`]({{< relref path="/ref/python/artifact#download" lang="ja" >}}) で可能なパラメータや返り値の種類について詳しく見てください。

{{% alert title="複数の組織に所属する個人エンティティを持つユーザー" %}} 
複数の組織に所属する個人エンティティを持つユーザーは、レジストリにリンクされたアーティファクトにアクセスする際、組織名を指定するか、チームエンティティを使用する必要があります。

```python
import wandb

REGISTRY = "<registry_name>"
COLLECTION = "<collection_name>"
VERSION = "<version>"

# API をインスタンス化する際に、自分のチームエンティティを使用していることを確認
api = wandb.Api(overrides={"entity": "<team-entity>"})
artifact_name = f"wandb-registry-{REGISTRY}/{COLLECTION}:{VERSION}"
artifact = api.artifact(name = artifact_name)

# パスに組織の表示名または組織エンティティを使用
api = wandb.Api()
artifact_name = f"{ORG_NAME}/wandb-registry-{REGISTRY}/{COLLECTION}:{VERSION}"
artifact = api.artifact(name = artifact_name)
```

`ORG_NAME` は組織の表示名です。マルチテナント SaaS ユーザーは、`https://wandb.ai/account-settings/` の組織の設定ページで組織名を見つけることができます。専用クラウドおよび自己管理ユーザーの場合、組織の表示名を確認するには、アカウント管理者に連絡してください。
{{% /alert %}}

## 事前に生成されたコードスニペットのコピーと貼り付け

W&B は、レジストリにリンクされたアーティファクトをダウンロードするために、Pythonスクリプト、ノートブック、またはターミナルにコピーして貼り付けることができるコードスニペットを作成します。

1. レジストリアプリに移動します。
2. アーティファクトを含むレジストリの名前を選択します。
3. コレクションの名前を選択します。
4. アーティファクトバージョンのリストからアクセスするバージョンを選択します。
5. **Usage** タブを選択します。
6. **Usage API** セクションに表示されたコードスニペットをコピーします。
7. コピーしたコードスニペットを Python スクリプト、ノートブック、またはターミナルに貼り付けます。

{{< img src="/images/registry/find_usage_in_registry_ui.gif" >}}