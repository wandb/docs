---
title: タグでバージョンを整理する
description: タグを使ってコレクション内やコレクション内の artifact バージョンを整理できます。タグの追加、削除、編集は Python SDK
  または W&B App の UI から行えます。
menu:
  default:
    identifier: ja-guides-core-registry-organize-with-tags
    parent: registry
weight: 7
---

コレクションやアーティファクトバージョンを整理するためにタグを作成し、コレクションやレジストリ内のアーティファクトバージョンへ追加しましょう。W&B App の UI または W&B Python SDK を使って、コレクションまたはアーティファクトバージョンにタグを追加、変更、表示、削除できます。

{{% alert title="タグとエイリアスの使い分け" %}}
特定のアーティファクトバージョンを一意に参照する必要がある場合は、エイリアスを使用します。例えば、`production` や `latest` のようなエイリアスを使えば、`artifact_name:alias` が常に特定バージョンだけを指すようになります。

一方で、より柔軟なグルーピングや検索をしたい場合にはタグを使います。複数のバージョンやコレクションが同じラベルを共有でき、特定の識別子に対して 1 つだけのバージョンと紐づく必要がない場合にはタグが最適です。
{{% /alert %}}


## コレクションにタグを追加する

W&B App UI または Python SDK を使って、コレクションにタグを追加できます。

{{< tabpane text=true >}}
{{% tab header="W&B App" %}}

W&B AppのUIでコレクションにタグを追加する方法:

1. [W&B Registry App](https://wandb.ai/registry) にアクセスします。
2. 任意のレジストリカードをクリックします。
3. コレクション名の横にある **View details** をクリックします。
4. コレクションカード内で、**Tags** フィールドの横にあるプラスアイコン（**+**）をクリックし、タグ名を入力します。
5. キーボードで **Enter** キーを押します。

{{< img src="/images/registry/add_tag_collection.gif" alt="レジストリコレクションへのタグ追加" >}}

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

`tags` 属性を再代入したり、書き換えることでプログラム上でタグを更新できます。W&B では、Python のベストプラクティスとして、`tags` 属性のインプレース更新ではなく再代入を推奨しています。

例えば、以下のコードスニペットは再代入によるタグリストの更新例です。簡潔さのため、[コレクションにタグを追加する]({{< relref path="#add-a-tag-to-a-collection" lang="ja" >}}) セクションのコード例を続けます。

```python
collection.tags = [*collection.tags, "new-tag", "other-tag"]
collection.tags = collection.tags + ["new-tag", "other-tag"]

collection.tags = set(collection.tags) - set(tags_to_delete)
collection.tags = []  # すべてのタグを削除
```

次のコードスニペットは、インプレース変更でコレクションのタグを更新する方法を示します。

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

W&B App UI を使って、コレクションに追加されたタグを確認できます。

1. [W&B Registry App](https://wandb.ai/registry) にアクセスします。
2. 任意のレジストリカードをクリックします。
3. コレクション名の横にある **View details** をクリックします。

コレクションに 1 つ以上のタグがある場合、コレクションカードの **Tags** フィールド横でそのタグを確認できます。

{{< img src="/images/registry/tag_collection_selected.png" alt="選択されたタグがあるレジストリコレクション" >}}

コレクションに追加されたタグは、コレクション名の横にも表示されます。

例えば、以下の画像では「tag1」というタグが「zoo-dataset-tensors」コレクションに追加されています。

{{< img src="/images/registry/tag_collection.png" alt="タグ管理" >}}


## コレクションからタグを削除する

W&B App UI を使って、コレクションからタグを削除できます。

1. [W&B Registry App](https://wandb.ai/registry) にアクセスします。
2. 任意のレジストリカードをクリックします。
3. コレクション名の横にある **View details** をクリックします。
4. コレクションカード内で、削除したいタグ名にカーソルを合わせます。
5. キャンセルボタン（**X** アイコン）をクリックします。

## アーティファクトバージョンにタグを追加する

コレクションに紐づくアーティファクトバージョンに、W&B App UI または Python SDK でタグを追加できます。

{{< tabpane text=true >}}
{{% tab header="W&B App" %}}
1. https://wandb.ai/registry で W&B Registry にアクセスします。
2. 任意のレジストリカードをクリックします。
3. タグを追加したいコレクション名の横にある **View details** をクリックします。
4. **Versions** までスクロールします。
5. アーティファクトバージョンの横の **View** をクリックします。
6. **Version** タブ内、**Tags** フィールドの横にあるプラスアイコン（**+**）をクリックし、タグ名を入力します。
7. キーボードで **Enter** キーを押します。

{{< img src="/images/registry/add_tag_linked_artifact_version.gif" alt="アーティファクトバージョンへのタグ追加" >}}

{{% /tab %}}
{{% tab header="Python SDK" %}}
タグを追加・更新したいアーティファクトバージョンを取得します。アーティファクトバージョンが取得できたら、そのオブジェクトの `tags` 属性にリスト形式で 1 つ以上のタグを代入してください。

他のアーティファクト同様、run を作成せずに W&B から直接取得することも、run 内でアーティファクトを取得することもできます。いずれの場合も、`save` メソッドを呼び出し、W&B サーバーに反映させてください。

以下の適切なコードをコピーし、`<>` の値を自身の環境に置き換えて利用してください。

新しい run を作成せずにアーティファクトを取得し、タグを追加する場合の例:

```python title="新しい run を作成せずにアーティファクトバージョンへタグを追加"
import wandb

ARTIFACT_TYPE = "<TYPE>"
ORG_NAME = "<org_name>"
REGISTRY_NAME = "<registry_name>"
COLLECTION_NAME = "<collection_name>"
VERSION = "<artifact_version>"

artifact_name = f"{ORG_NAME}/wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}:v{VERSION}"

artifact = wandb.Api().artifact(name = artifact_name, type = ARTIFACT_TYPE)
artifact.tags = ["tag2"] # リストとしてタグを指定
artifact.save()
```


新しい run を作成し、タグを追加する例:

```python title="run 実行中にアーティファクトバージョンへタグを追加"
import wandb

ORG_NAME = "<org_name>"
REGISTRY_NAME = "<registry_name>"
COLLECTION_NAME = "<collection_name>"
VERSION = "<artifact_version>"

run = wandb.init(entity = "<entity>", project="<project>")

artifact_name = f"{ORG_NAME}/wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}:v{VERSION}"

artifact = run.use_artifact(artifact_or_name = artifact_name)
artifact.tags = ["tag2"] # リストとしてタグを指定
artifact.save()
```

{{% /tab %}}
{{< /tabpane >}}



## アーティファクトバージョンのタグを更新する

`tags` 属性を再代入したり、書き換えることで、プログラム上でタグを更新できます。W&B では、Python のベストプラクティスとして `tags` 属性のインプレース変更よりも再代入を推奨します。

例えば、以下のコードスニペットは再代入によるタグリストの更新例です。簡潔さのため、[アーティファクトバージョンにタグを追加する]({{< relref path="#add-a-tag-to-an-artifact-version" lang="ja" >}}) セクションのコード例を続けます。

```python
artifact.tags = [*artifact.tags, "new-tag", "other-tag"]
artifact.tags = artifact.tags + ["new-tag", "other-tag"]

artifact.tags = set(artifact.tags) - set(tags_to_delete)
artifact.tags = []  # すべてのタグを削除
```

次のコードスニペットは、インプレース変更でアーティファクトバージョンのタグを更新する方法です。

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

レジストリに紐づいたアーティファクトバージョンのタグを、W&B App UI または Python SDK で表示できます。

{{< tabpane text=true >}}
{{% tab header="W&B App" %}}

1. [W&B Registry App](https://wandb.ai/registry) にアクセスします。
2. 任意のレジストリカードをクリックします。
3. タグを追加したいコレクション名の横にある **View details** をクリックします。
4. **Versions** セクションまでスクロールします。

アーティファクトバージョンに 1 つ以上のタグがある場合、**Tags** カラム内で確認できます。

{{< img src="/images/registry/tag_artifact_version.png" alt="タグ付きアーティファクトバージョン" >}}

{{% /tab %}}
{{% tab header="Python SDK" %}}

表示したいアーティファクトバージョンを取得してください。オブジェクトの `tags` 属性を参照することで、そのアーティファクトのタグを確認できます。

他のアーティファクト同様、run を作成せずに W&B から直接取得することも、run 内でアーティファクトを取得することもできます。

以下の適切なコードをコピーし、`<>` の値を自身の環境に置き換えて利用してください。

新しい run を作成せずにアーティファクトバージョンのタグを表示する例:

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


新しい run を作成し、アーティファクトバージョンのタグを表示する例:

```python title="run 実行中にアーティファクトバージョンのタグを表示"
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
2. 任意のレジストリカードをクリックします。
3. タグを追加したいコレクション名の横にある **View details** をクリックします。
4. **Versions** までスクロールします。
5. アーティファクトバージョンの横の **View** をクリックします。
6. **Version** タブ内で、タグ名にカーソルを合わせます。
7. キャンセルボタン（**X** アイコン）をクリックします。

## 既存タグの検索

W&B App UI では、コレクションやアーティファクトバージョンに存在するタグを検索できます。

1. [W&B Registry App](https://wandb.ai/registry) にアクセスします。
2. 任意のレジストリカードをクリックします。
3. 検索バーでタグ名を入力します。

{{< img src="/images/registry/search_tags.gif" alt="タグ検索" >}}


## 特定のタグを持つアーティファクトバージョンの検索

特定のタグを持つアーティファクトバージョンを W&B Python SDK で検索できます。

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