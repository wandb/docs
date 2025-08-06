---
title: レジストリからアーティファクトをダウンロードする
menu:
  default:
    identifier: download_use_artifact
    parent: registry
weight: 6
---

W&B Python SDK を使って、レジストリにリンクされたアーティファクトをダウンロードできます。アーティファクトをダウンロードして利用するには、レジストリ名、コレクション名、そしてダウンロードしたいアーティファクトバージョンのエイリアスまたはインデックスを知っている必要があります。

アーティファクトのプロパティがわかったら、[リンクされたアーティファクトへのパスを構築する]({{< relref "#construct-path-to-linked-artifact" >}})ことでダウンロードできます。あるいは、W&B アプリ UI で事前に生成された[コードスニペットをコピー＆ペースト]({{< relref "#copy-and-paste-pre-generated-code-snippet" >}})して、レジストリにリンクされたアーティファクトをダウンロードすることもできます。

## リンクされたアーティファクトへのパスを構築する

レジストリにリンクされたアーティファクトをダウンロードするには、そのリンクされたアーティファクトのパスを知っている必要があります。このパスは、レジストリ名、コレクション名、そしてアクセスしたいアーティファクトバージョンのエイリアスまたはインデックスで構成されています。

レジストリ、コレクション、そしてアーティファクトバージョンのエイリアスまたはインデックスが分かったら、以下の文字列テンプレートを使ってリンクされたアーティファクトへのパスを構築できます。

```python
# バージョンインデックス指定の場合
f"wandb-registry-{REGISTRY}/{COLLECTION}:v{INDEX}"

# エイリアス指定の場合
f"wandb-registry-{REGISTRY}/{COLLECTION}:{ALIAS}"
```

中括弧 `{}` 内の値を、アクセスしたいレジストリ名・コレクション名・アーティファクトバージョンのエイリアスまたはインデックスに置き換えてください。

{{% alert %}}
アーティファクトバージョンを core Model registry や core Dataset registry にリンクする場合は、それぞれ `model` や `dataset` を指定してください。
{{% /alert %}}

リンクされたアーティファクトのパスが分かったら、`wandb.init.use_artifact` メソッドを使ってアーティファクトにアクセスし、その内容をダウンロードできます。以下のコードスニペットは、W&B Registry にリンクされたアーティファクトの利用・ダウンロード例です。`<>` 内の値はご自身のものに置き換えてください。

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
# artifact_name = '<artifact_name>' # Registry App で指定されたフルネームをコピー＆ペーストして利用可
fetched_artifact = run.use_artifact(artifact_or_name = artifact_name)  
download_path = fetched_artifact.download()  
```

`.use_artifact()` メソッドは [run]({{< relref "/guides/models/track/runs/" >}}) を作成すると同時に、ダウンロードしたアーティファクトをその run の入力としてマークします。
アーティファクトを run の入力としてマークすることで、W&B はそのアーティファクトのリネージ追跡を行えるようになります。

もし run を作成したくない場合は、`wandb.Api()` オブジェクトを利用してアーティファクトへアクセスできます。

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
<summary>例： W&B Registry にリンクされたアーティファクトを利用＆ダウンロードする</summary>

以下のコード例は、ユーザーが **Fine-tuned Models** registry の `phi3-finetuned` というコレクションにリンクされたアーティファクトをダウンロードする方法を示します。アーティファクトバージョンのエイリアスは `production` に設定されています。

```python
import wandb

TEAM_ENTITY = "product-team-applications"
PROJECT_NAME = "user-stories"

REGISTRY = "Fine-tuned Models"
COLLECTION = "phi3-finetuned"
ALIAS = 'production'

# 指定したチームとプロジェクト内で run を初期化
run = wandb.init(entity=TEAM_ENTITY, project = PROJECT_NAME)

artifact_name = f"wandb-registry-{REGISTRY}/{COLLECTION}:{ALIAS}"

# アーティファクトへアクセスし、run の入力としてリネージ追跡のためにマーク
fetched_artifact = run.use_artifact(artifact_or_name = name)  

# アーティファクトをダウンロード。ダウンロードした内容へのパスが返る
downloaded_path = fetched_artifact.download()  
```
</details>

パラメータや戻り値の型については、APIリファレンスの [`use_artifact`]({{< relref "/ref/python/sdk/classes/run.md#use_artifact" >}}) および [`Artifact.download()`]({{< relref "/ref/python/sdk/classes/artifact.md#download" >}}) を参照してください。

{{% alert title="複数の組織に所属する個人エンティティを持つユーザーの場合" %}} 
複数の組織に所属する個人エンティティを持つユーザーが、レジストリにリンクされたアーティファクトにアクセスする場合は、組織名またはチームエンティティ名いずれかをパスで指定する必要があります。

```python
import wandb

REGISTRY = "<registry_name>"
COLLECTION = "<collection_name>"
VERSION = "<version>"

# APIインスタンス作成時に team entity を指定
api = wandb.Api(overrides={"entity": "<team-entity>"})
artifact_name = f"wandb-registry-{REGISTRY}/{COLLECTION}:{VERSION}"
artifact = api.artifact(name = artifact_name)

# パスで org display name または org entity を利用
api = wandb.Api()
artifact_name = f"{ORG_NAME}/wandb-registry-{REGISTRY}/{COLLECTION}:{VERSION}"
artifact = api.artifact(name = artifact_name)
```

`ORG_NAME` には組織の表示名を指定します。マルチテナント SaaS ユーザーの場合、組織の表示名は `https://wandb.ai/account-settings/` の組織設定ページで確認できます。専用クラウドおよびセルフマネージド環境のユーザーは、組織の表示名を担当管理者にご確認ください。
{{% /alert %}}

## 事前生成されたコードスニペットをコピー＆ペースト

W&B では、レジストリにリンクされたアーティファクトをダウンロードするためのコードスニペットが生成されており、Python スクリプト、ノートブック、あるいはターミナルにそのままコピー＆ペーストできます。

1. Registry App へ移動します。
2. アーティファクトが含まれるレジストリ名を選択します。
3. コレクション名を選択します。
4. アーティファクトバージョンの一覧から、アクセスしたいバージョンを選択します。
5. **Usage** タブを選択します。
6. **Usage API** セクションに表示されているコードスニペットをコピーします。
7. そのコードスニペットを Python スクリプト、ノートブック、またはターミナルに貼り付けます。

{{< img src="/images/registry/find_usage_in_registry_ui.gif" >}}