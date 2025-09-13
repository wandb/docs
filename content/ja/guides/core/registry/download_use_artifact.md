---
title: レジストリから Artifacts をダウンロード
menu:
  default:
    identifier: ja-guides-core-registry-download_use_artifact
    parent: registry
weight: 6
---

W&B Python SDK を使って、レジストリにリンクされた artifact をダウンロードします。artifact をダウンロードして使うには、レジストリ名、コレクション名、そしてダウンロードしたい artifact バージョンのエイリアスまたはインデックスを把握している必要があります。

artifact のプロパティが分かっていれば、[リンクされた artifact へのパスを構築し]({{< relref path="#construct-path-to-linked-artifact" lang="ja" >}}) artifact をダウンロードできます。あるいは、W&B App の UI から[事前生成されたコードスニペットをコピー＆ペーストし]({{< relref path="#copy-and-paste-pre-generated-code-snippet" lang="ja" >}})、レジストリにリンクされた artifact をダウンロードすることもできます。

## リンクされた artifact へのパスを構築する

レジストリにリンクされた artifact をダウンロードするには、そのリンク先の artifact のパスを把握しておく必要があります。このパスは、レジストリ名、コレクション名、アクセスしたい artifact バージョンのエイリアスまたはインデックスで構成されます。

artifact バージョンのレジストリ、コレクション、エイリアスまたはインデックスが分かっていれば、次の文字列テンプレートでリンクされた artifact へのパスを組み立てられます。

```python
# バージョンインデックスを指定した artifact 名
f"wandb-registry-{REGISTRY}/{COLLECTION}:v{INDEX}"

# エイリアスを指定した artifact 名
f"wandb-registry-{REGISTRY}/{COLLECTION}:{ALIAS}"
```

波括弧 `{}` 内の値を、アクセスしたいレジストリ名、コレクション名、artifact バージョンのエイリアスまたはインデックスに置き換えてください。

{{% alert %}}
artifact バージョンをそれぞれコア モデルレジストリ または コア データセット レジストリにリンクするには、`model` または `dataset` を指定します。
{{% /alert %}}

リンクされた artifact のパスが分かったら、`wandb.init.use_artifact` メソッドを使って artifact にアクセスし、内容をダウンロードします。次のコードスニペットは、W&B Registry にリンクされた artifact を使ってダウンロードする方法を示しています。`<>` 内の値はご自身のものに置き換えてください。

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
# Registry App で指定された完全名をコピー＆ペーストします
fetched_artifact = run.use_artifact(artifact_or_name = artifact_name)  
download_path = fetched_artifact.download()  
```

`.use_artifact()` メソッドは、[Run]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) を作成し、ダウンロードした artifact をその Run の入力としてマークします。artifact を Run の入力としてマークすることで、W&B はその artifact のリネージを追跡できます。

Run を作成したくない場合は、`wandb.Api()` オブジェクトを使用して artifact にアクセスできます。

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
<summary>例: W&B Registry にリンクされた artifact を使ってダウンロードする</summary>

以下のコード例は、**Fine-tuned Models** レジストリ内の `phi3-finetuned` というコレクションにリンクされた artifact を User がダウンロードする方法を示しています。artifact バージョンのエイリアスは `production` に設定されています。

```python
import wandb

TEAM_ENTITY = "product-team-applications"
PROJECT_NAME = "user-stories"

REGISTRY = "Fine-tuned Models"
COLLECTION = "phi3-finetuned"
ALIAS = 'production'

# 指定された Team と Project 内で Run を初期化します
run = wandb.init(entity=TEAM_ENTITY, project = PROJECT_NAME)

artifact_name = f"wandb-registry-{REGISTRY}/{COLLECTION}:{ALIAS}"

# artifact にアクセスし、リネージ追跡のために Run の入力としてマークします
fetched_artifact = run.use_artifact(artifact_or_name = name)  

# artifact をダウンロードします。ダウンロードされた内容へのパスを返します
downloaded_path = fetched_artifact.download()  
```
</details>

パラメータや戻り値の型については、API リファレンスの [`use_artifact`]({{< relref path="/ref/python/sdk/classes/run.md#use_artifact" lang="ja" >}}) および [`Artifact.download()`]({{< relref path="/ref/python/sdk/classes/artifact.md#download" lang="ja" >}}) を参照してください。

{{% alert title="複数の組織に所属する個人の Entity を持つ User" %}}
複数の組織に所属する個人の Entity を持つ User は、レジストリにリンクされた artifact にアクセスする際に、組織の名前または Team Entity のいずれかを指定する必要があります。

```python
import wandb

REGISTRY = "<registry_name>"
COLLECTION = "<collection_name>"
VERSION = "<version>"

# Team Entity を使って API をインスタンス化していることを確認してください
api = wandb.Api(overrides={"entity": "<team-entity>"})
artifact_name = f"wandb-registry-{REGISTRY}/{COLLECTION}:{VERSION}"
artifact = api.artifact(name = artifact_name)

# パスでは組織の表示名または組織の Entity を使用します
api = wandb.Api()
artifact_name = f"{ORG_NAME}/wandb-registry-{REGISTRY}/{COLLECTION}:{VERSION}"
artifact = api.artifact(name = artifact_name)
```

ここで `ORG_NAME` は組織の表示名です。マルチテナント SaaS の User は、組織の設定ページ (`https://wandb.ai/account-settings/`) で組織の名前を確認できます。専用クラウド および セルフマネージド の User は、アカウント管理者に連絡して組織の表示名を確認してください。
{{% /alert %}}

## 事前生成されたコードスニペットをコピー＆ペーストする

W&B は、レジストリにリンクされた artifact をダウンロードするために、Python スクリプト、ノートブック、またはターミナルにコピー＆ペーストできるコードスニペットを生成します。

1. Registry App に移動します。
2. artifact を含むレジストリの名前を選択します。
3. コレクション名を選択します。
4. artifact バージョンのリストから、アクセスしたいバージョンを選択します。
5. **Usage** タブを選択します。
6. **Usage API** セクションに表示されているコードスニペットをコピーします。
7. コードスニペットを Python スクリプト、ノートブック、またはターミナルに貼り付けます。

{{< img src="/images/registry/find_usage_in_registry_ui.gif" >}}