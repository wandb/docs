---
title: タグでバージョンを整理する
description: タグを使ってコレクション内やコレクション内の artifact バージョンを整理できます。タグの追加、削除、編集は Python SDK
  や W&B App UI から行えます。
menu:
  default:
    identifier: organize-with-tags
    parent: registry
weight: 7
---

コレクションやアーティファクトバージョンを整理するためにタグを作成して追加し、レジストリ内で管理しましょう。タグの追加、編集、表示、削除は W&B App UI または W&B Python SDK から行えます。

{{% alert title="タグとエイリアスの使い分けについて" %}}
特定のアーティファクトバージョンを一意に参照したい場合はエイリアスを使いましょう。たとえば `production` や `latest` のようなエイリアスを使うことで、`artifact_name:alias` が常に特定のバージョンだけを指すことを保証できます。

一方、より柔軟にグループ化や検索をしたい場合はタグが便利です。複数のバージョンやコレクションが同じラベルを共有でき、特定の識別子にバージョンが限定される必要がないケースに適しています。
{{% /alert %}}


## コレクションにタグを追加する

W&B App UI または Python SDK を使ってコレクションにタグを追加できます。

{{< tabpane text=true >}}
{{% tab header="W&B App" %}}

W&B App UI でコレクションにタグを追加する手順：

1. [W&B Registry App](https://wandb.ai/registry) にアクセスします。
2. レジストリカードをクリックします
3. コレクション名の横にある **View details** をクリックします
4. コレクションカード内で、**Tags** 項目の横のプラスアイコン（**+**）をクリックし、タグ名を入力します
5. キーボードで **Enter** を押します

{{< img src="/images/registry/add_tag_collection.gif" alt="Registry コレクションへのタグ追加" >}}

{{% /tab %}}
{{% tab header="Python SDK" %}}

```python
import wandb

COLLECTION_TYPE = "<collection_type>"
ORG_NAME = "<org_name>"
REGISTRY_NAME = "<registry_name>"
COLLECTION_NAME = "<collection_name>"

full_name = f"{ORG_NAME}/wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}"

collection = wandb.Api().artifact_collection(
  type_name = COLLECTION_TYPE, 
  name = full_name
  )

collection.tags = ["your-tag"]
collection.save()
```

{{% /tab %}}
{{< /tabpane >}}



## コレクションのタグを更新する

プログラムでタグを更新するには `tags` 属性を再代入または直接操作します。W&B では Python の推奨パターンとして、インプレース変更ではなく再代入を推奨しています。

例えば、以下のコードスニペットでは再代入による list の更新方法を例示しています。詳細は [コレクションにタグを追加する]({{< relref "#add-a-tag-to-a-collection" >}})のコード例を参照してください。

```python
collection.tags = [*collection.tags, "new-tag", "other-tag"]
collection.tags = collection.tags + ["new-tag", "other-tag"]

collection.tags = set(collection.tags) - set(tags_to_delete)
collection.tags = []  # すべてのタグを削除
```

次のコードスニペットはインプレース変更によってタグを更新する例です。

```python
collection.tags += ["new-tag", "other-tag"]
collection.tags.append("new-tag")

collection.tags.extend(["new-tag", "other-tag"])
collection.tags[:] = ["new-tag", "other-tag"]
collection.tags.remove("existing-tag")
collection.tags.pop()
collection.tags.clear()
```

## コレクションのタグを表示する

W&B App UI でコレクションに追加されたタグを表示できます。

1. [W&B Registry App](https://wandb.ai/registry) にアクセスします。
2. レジストリカードをクリックします
3. コレクション名の横にある **View details** をクリックします

コレクションに1つ以上タグがある場合、コレクションカード内の **Tags** 項目の横で確認できます。

{{< img src="/images/registry/tag_collection_selected.png" alt="タグが選択された Registry コレクション" >}}

コレクションに追加されたタグは、コレクション名の横にも表示されます。

例えば、次の画像では「tag1」というタグが "zoo-dataset-tensors" コレクションに追加されています。

{{< img src="/images/registry/tag_collection.png" alt="タグ管理" >}}


## コレクションからタグを削除する

W&B App UI を使ってコレクションからタグを削除する方法：

1. [W&B Registry App](https://wandb.ai/registry) にアクセスします。
2. レジストリカードをクリックします
3. コレクション名の横にある **View details** をクリックします
4. コレクションカード内で、削除したいタグ名にマウスポインタを合わせます
5. キャンセルボタン（**X** アイコン）をクリックします

## アーティファクトバージョンにタグを追加する

コレクションに紐づくアーティファクトバージョンにタグを追加するには、W&B App UI または Python SDK を利用します。

{{< tabpane text=true >}}
{{% tab header="W&B App" %}}
1. W&B Registry（https://wandb.ai/registry）にアクセスします
2. レジストリカードをクリックします
3. タグを追加したいコレクション名の横にある **View details** をクリックします
4. **Versions** までスクロールします
5. アーティファクトバージョンの横にある **View** をクリックします
6. **Version** タブ内で、**Tags** 項目の横のプラスアイコン（**+**）をクリックし、タグ名を入力します
7. キーボードで **Enter** を押します

{{< img src="/images/registry/add_tag_linked_artifact_version.gif" alt="アーティファクトバージョンへのタグ追加" >}}

{{% /tab %}}
{{% tab header="Python SDK" %}}
タグを追加したいアーティファクトバージョンを取得し、オブジェクトの `tag` 属性にリストでタグを指定します。

他の Artifacts 同様、run を作成せずに取得することも、run 作成後に取得することもできます。いずれの場合も `save` メソッドを呼び出して W&B サーバー上でアーティファクトの更新を反映します。

適切なセルをコピー＆ペーストし、`<>` 内はご自身の値に置き換えてください。

新しい run を作成せずにアーティファクトへタグを追加する例：

```python title="新しい run を作成せずにアーティファクトバージョンにタグを追加"
import wandb

ARTIFACT_TYPE = "<TYPE>"
ORG_NAME = "<org_name>"
REGISTRY_NAME = "<registry_name>"
COLLECTION_NAME = "<collection_name>"
VERSION = "<artifact_version>"

artifact_name = f"{ORG_NAME}/wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}:v{VERSION}"

artifact = wandb.Api().artifact(name = artifact_name, type = ARTIFACT_TYPE)
artifact.tags = ["tag2"] # リストで1つ以上のタグを指定
artifact.save()
```


新しい run を作成してアーティファクトへタグを追加する例：

```python title="run 中にアーティファクトバージョンにタグを追加"
import wandb

ORG_NAME = "<org_name>"
REGISTRY_NAME = "<registry_name>"
COLLECTION_NAME = "<collection_name>"
VERSION = "<artifact_version>"

run = wandb.init(entity = "<entity>", project="<project>")

artifact_name = f"{ORG_NAME}/wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}:v{VERSION}"

artifact = run.use_artifact(artifact_or_name = artifact_name)
artifact.tags = ["tag2"] # リストで1つ以上のタグを指定
artifact.save()
```

{{% /tab %}}
{{< /tabpane >}}



## アーティファクトバージョンのタグを更新する

プログラムでタグを更新するには、`tags` 属性を再代入または直接操作します。W&B では、インプレース変更ではなく、属性の再代入を推奨しています。

例えば、以下のコードスニペットは [アーティファクトバージョンにタグを追加する]({{< relref "#add-a-tag-to-an-artifact-version" >}})の例からの続きです。

```python
artifact.tags = [*artifact.tags, "new-tag", "other-tag"]
artifact.tags = artifact.tags + ["new-tag", "other-tag"]

artifact.tags = set(artifact.tags) - set(tags_to_delete)
artifact.tags = []  # すべてのタグを削除
```

次に、インプレース変更によるタグ更新例です。

```python
artifact.tags += ["new-tag", "other-tag"]
artifact.tags.append("new-tag")

artifact.tags.extend(["new-tag", "other-tag"])
artifact.tags[:] = ["new-tag", "other-tag"]
artifact.tags.remove("existing-tag")
artifact.tags.pop()
artifact.tags.clear()
```


## アーティファクトバージョンのタグを表示する

W&B App UI または Python SDK を使用して、レジストリに紐づくアーティファクトバージョンのタグを確認できます。

{{< tabpane text=true >}}
{{% tab header="W&B App" %}}

1. [W&B Registry App](https://wandb.ai/registry) にアクセスします。
2. レジストリカードをクリックします
3. タグを追加したいコレクション名の横にある **View details** をクリックします
4. **Versions** セクションまでスクロールします

アーティファクトバージョンに1つ以上タグが付与されていれば、**Tags** カラムで確認できます。

{{< img src="/images/registry/tag_artifact_version.png" alt="タグ付きアーティファクトバージョン" >}}

{{% /tab %}}
{{% tab header="Python SDK" %}}

タグを確認したいアーティファクトバージョンを取得し、オブジェクトの `tag` 属性でタグを確認できます。

他の Artifacts 同様、run を作成せずに取得したり、run 内で取得することが可能です。

適切なセルをコピー＆ペーストし、`<>` 内はご自身の値に置き換えてください。

新しい run を作成せずにアーティファクトのタグを取得する例：

```python title="新しい run を作成せずにアーティファクトバージョンのタグを表示"
import wandb

ARTIFACT_TYPE = "<TYPE>"
ORG_NAME = "<org_name>"
REGISTRY_NAME = "<registry_name>"
COLLECTION_NAME = "<collection_name>"
VERSION = "<artifact_version>"

artifact_name = f"{ORG_NAME}/wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}:v{VERSION}"

artifact = wandb.Api().artifact(name = artifact_name, type = artifact_type)
print(artifact.tags)
```


run 中にアーティファクトバージョンのタグを取得する例：

```python title="run 中にアーティファクトバージョンのタグを表示"
import wandb

ORG_NAME = "<org_name>"
REGISTRY_NAME = "<registry_name>"
COLLECTION_NAME = "<collection_name>"
VERSION = "<artifact_version>"

run = wandb.init(entity = "<entity>", project="<project>")

artifact_name = f"{ORG_NAME}/wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}:v{VERSION}"

artifact = run.use_artifact(artifact_or_name = artifact_name)
print(artifact.tags)
```

{{% /tab %}}
{{< /tabpane >}}



## アーティファクトバージョンからタグを削除する

1. [W&B Registry App](https://wandb.ai/registry) にアクセスします。
2. レジストリカードをクリックします
3. タグを追加したいコレクション名の横にある **View details** をクリックします
4. **Versions** にスクロールします
5. アーティファクトバージョンの横にある **View** をクリックします
6. **Version** タブ内で、タグ名にマウスオーバーします
7. キャンセルボタン（**X** アイコン）をクリックします

## 既存タグの検索

コレクションやアーティファクトバージョン内で、W&B App UI の検索バーから既存のタグを検索できます。

1. [W&B Registry App](https://wandb.ai/registry) にアクセスします。
2. レジストリカードをクリックします
3. 検索バーにタグ名を入力します。

{{< img src="/images/registry/search_tags.gif" alt="タグ検索" >}}


## 特定のタグを持つアーティファクトバージョンを探す

W&B Python SDK で指定タグを持つアーティファクトバージョンを検索できます：

```python
import wandb

api = wandb.Api()
tagged_artifact_versions = api.artifacts(
    type_name = "<artifact_type>",
    name = "<artifact_name>",
    tags = ["<tag_1>", "<tag_2>"]
)

for artifact_version in tagged_artifact_versions:
    print(artifact_version.tags)
```