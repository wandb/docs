---
title: タグでバージョンを整理する
description: タグを使用して、コレクションやコレクション内の Artifacts のバージョンを整理します。Python SDK または W&B App
  UI を使用して、タグの追加、削除、編集が可能です。
menu:
  default:
    identifier: ja-guides-core-registry-organize-with-tags
    parent: registry
weight: 7
---

レジストリ内でコレクションやアーティファクト バージョンを整理するために、タグを作成して追加します。W&B App UI または W&B Python SDK を使用して、コレクションやアーティファクト バージョンにタグを追加、変更、表示、または削除できます。

{{% alert title="タグとエイリアスの使い分け" %}}
特定のアーティファクト バージョンを一意に参照する必要がある場合は、エイリアスを使用します。たとえば、「production」や「latest」などのエイリアスを使用して、`artifact_name:alias` が常に単一の特定のバージョンを指すようにします。

グループ化や検索の柔軟性を高めたい場合は、タグを使用します。複数のバージョンまたはコレクションが同じラベルを共有でき、単一のバージョンが特定の識別子に関連付けられている保証が必要ない場合に、タグは理想的です。
{{% /alert %}}


## コレクションにタグを追加する

W&B App UI または Python SDK を使用して、コレクションにタグを追加します。

{{< tabpane text=true >}}
{{% tab header="W&B App" %}}

W&B App UI を使用して、コレクションにタグを追加します。

1. [W&B Registry App](https://wandb.ai/registry) に移動します。
2. レジストリ カードをクリックします。
3. コレクション名の横にある **View details** をクリックします。
4. コレクション カード内で、**Tags** フィールドの横にあるプラスアイコン (**+**) をクリックし、タグの名前を入力します。
5. キーボードの **Enter** を押します。

{{< img src="/images/registry/add_tag_collection.gif" alt="レジストリ コレクションにタグを追加する" >}}

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



## コレクションに属するタグを更新する

`tags` 属性を再割り当てするか、変更することによって、プログラムでタグを更新します。W&B では、インプレース変更の代わりに `tags` 属性を再割り当てすることを推奨しており、これは良い Python の習慣です。

たとえば、以下のコードスニペットは、再割り当てでリストを更新する一般的な方法を示しています。簡潔にするため、[「コレクションにタグを追加する」セクション]({{< relref path="#add-a-tag-to-a-collection" lang="ja" >}}) のコード例を続行します。

```python
collection.tags = [*collection.tags, "new-tag", "other-tag"]
collection.tags = collection.tags + ["new-tag", "other-tag"]

collection.tags = set(collection.tags) - set(tags_to_delete)
collection.tags = []  # deletes all tags
```

以下のコードスニペットは、インプレース変更を使用してコレクションに属するタグを更新する方法を示しています。

```python
collection.tags += ["new-tag", "other-tag"]
collection.tags.append("new-tag")

collection.tags.extend(["new-tag", "other-tag"])
collection.tags[:] = ["new-tag", "other-tag"]
collection.tags.remove("existing-tag")
collection.tags.pop()
collection.tags.clear()
```

## コレクションに属するタグを表示する

W&B App UI を使用して、コレクションに追加されたタグを表示します。

1. [W&B Registry App](https://wandb.ai/registry) に移動します。
2. レジストリ カードをクリックします。
3. コレクション名の横にある **View details** をクリックします。

コレクションに 1 つ以上のタグがある場合、コレクション カード内の **Tags** フィールドの横にそれらのタグが表示されます。

{{< img src="/images/registry/tag_collection_selected.png" alt="選択されたタグがあるレジストリ コレクション" >}}

コレクションに追加されたタグは、そのコレクション名の横にも表示されます。

たとえば、以下の画像では、「tag1」というタグが「zoo-dataset-tensors」コレクションに追加されています。

{{< img src="/images/registry/tag_collection.png" alt="タグ管理" >}}


## コレクションからタグを削除する

W&B App UI を使用して、コレクションからタグを削除します。

1. [W&B Registry App](https://wandb.ai/registry) に移動します。
2. レジストリ カードをクリックします。
3. タグを削除したいコレクション名の横にある **View details** をクリックします。
4. コレクション カード内で、削除したいタグ名にマウスを合わせます。
5. キャンセル ボタン (**X** アイコン) をクリックします。

## アーティファクト バージョンにタグを追加する

W&B App UI または Python SDK を使用して、コレクションにリンクされたアーティファクト バージョンにタグを追加します。

{{< tabpane text=true >}}
{{% tab header="W&B App" %}}
1. https://wandb.ai/registry の W&B Registry に移動します。
2. レジストリ カードをクリックします。
3. タグを追加したいコレクション名の横にある **View details** をクリックします。
4. **Versions** までスクロールします。
5. アーティファクト バージョンの横にある **View** をクリックします。
6. **Version** タブ内で、**Tags** フィールドの横にあるプラスアイコン (**+**) をクリックし、タグの名前を入力します。
7. キーボードの **Enter** を押します。

{{< img src="/images/registry/add_tag_linked_artifact_version.gif" alt="アーティファクト バージョンにタグを追加する" >}}

{{% /tab %}}
{{% tab header="Python SDK" %}}
タグを追加または更新したいアーティファクト バージョンをフェッチします。アーティファクト バージョンを取得したら、アーティファクト オブジェクトの `tags` 属性にアクセスして、そのアーティファクトにタグを追加または変更できます。1 つ以上のタグをリストとしてアーティファクトの `tags` 属性に渡します。

他のアーティファクトと同様に、run を作成せずに W&B からアーティファクトをフェッチすることも、run を作成してその run 内でアーティファクトをフェッチすることもできます。どちらの場合も、W&B サーバー上のアーティファクトを更新するために、アーティファクト オブジェクトの `save` メソッドを呼び出すようにしてください。

以下から適切なコードセルをコピー＆ペーストして、アーティファクト バージョンのタグを追加または変更します。`< >` 内の値を自分の値に置き換えてください。


以下のコードスニペットは、新しい run を作成せずにアーティファクトをフェッチしてタグを追加する方法を示しています。
```python title="新しい run を作成せずにアーティファクト バージョンにタグを追加する"
import wandb

ARTIFACT_TYPE = "<TYPE>"
ORG_NAME = "<org_name>"
REGISTRY_NAME = "<registry_name>"
COLLECTION_NAME = "<collection_name>"
VERSION = "<artifact_version>"

artifact_name = f"{ORG_NAME}/wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}:v{VERSION}"

artifact = wandb.Api().artifact(name = artifact_name, type = ARTIFACT_TYPE)
artifact.tags = ["tag2"] # リストに 1 つ以上のタグを指定します
artifact.save()
```


以下のコードスニペットは、新しい run を作成してアーティファクトをフェッチし、タグを追加する方法を示しています。

```python title="run の実行中にアーティファクト バージョンにタグを追加する"
import wandb

ORG_NAME = "<org_name>"
REGISTRY_NAME = "<registry_name>"
COLLECTION_NAME = "<collection_name>"
VERSION = "<artifact_version>"

run = wandb.init(entity = "<entity>", project="<project>")

artifact_name = f"{ORG_NAME}/wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}:v{VERSION}"

artifact = run.use_artifact(artifact_or_name = artifact_name)
artifact.tags = ["tag2"] # リストに 1 つ以上のタグを指定します
artifact.save()
```

{{% /tab %}}
{{< /tabpane >}}



## アーティファクト バージョンに属するタグを更新する


`tags` 属性を再割り当てするか、変更することによって、プログラムでタグを更新します。W&B では、インプレース変更の代わりに `tags` 属性を再割り当てすることを推奨しており、これは良い Python の習慣です。

たとえば、以下のコードスニペットは、再割り当てでリストを更新する一般的な方法を示しています。簡潔にするため、[「アーティファクト バージョンにタグを追加する」セクション]({{< relref path="#add-a-tag-to-an-artifact-version" lang="ja" >}}) のコード例を続行します。

```python
artifact.tags = [*artifact.tags, "new-tag", "other-tag"]
artifact.tags = artifact.tags + ["new-tag", "other-tag"]

artifact.tags = set(artifact.tags) - set(tags_to_delete)
artifact.tags = []  # deletes all tags
```

以下のコードスニペットは、インプレース変更を使用してアーティファクト バージョンに属するタグを更新する方法を示しています。

```python
artifact.tags += ["new-tag", "other-tag"]
artifact.tags.append("new-tag")

artifact.tags.extend(["new-tag", "other-tag"])
artifact.tags[:] = ["new-tag", "other-tag"]
artifact.tags.remove("existing-tag")
artifact.tags.pop()
artifact.tags.clear()
```


## アーティファクト バージョンに属するタグを表示する

W&B App UI または Python SDK を使用して、レジストリにリンクされたアーティファクト バージョンに属するタグを表示します。

{{< tabpane text=true >}}
{{% tab header="W&B App" %}}

1. [W&B Registry App](https://wandb.ai/registry) に移動します。
2. レジストリ カードをクリックします。
3. タグを追加したいコレクション名の横にある **View details** をクリックします。
4. **Versions** セクションまでスクロールします。

アーティファクト バージョンに 1 つ以上のタグがある場合、**Tags** 列にそれらのタグが表示されます。

{{< img src="/images/registry/tag_artifact_version.png" alt="タグ付きアーティファクト バージョン" >}}

{{% /tab %}}
{{% tab header="Python SDK" %}}

アーティファクト バージョンをフェッチして、そのタグを表示します。アーティファクト バージョンを取得したら、アーティファクト オブジェクトの `tags` 属性を表示することで、そのアーティファクトに属するタグを確認できます。

他のアーティファクトと同様に、run を作成せずに W&B からアーティファクトをフェッチすることも、run を作成してその run 内でアーティファクトをフェッチすることもできます。

以下から適切なコードセルをコピー＆ペーストして、アーティファクト バージョンのタグを表示します。`< >` 内の値を自分の値に置き換えてください。

以下のコードスニペットは、新しい run を作成せずにアーティファクト バージョンのタグをフェッチして表示する方法を示しています。

```python title="新しい run を作成せずにアーティファクト バージョンのタグを表示する"
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


以下のコードスニペットは、新しい run を作成してアーティファクト バージョンのタグをフェッチして表示する方法を示しています。

```python title="run の実行中にアーティファクト バージョンのタグを表示する"
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



## アーティファクト バージョンからタグを削除する

1. [W&B Registry App](https://wandb.ai/registry) に移動します。
2. レジストリ カードをクリックします。
3. タグを削除したいコレクション名の横にある **View details** をクリックします。
4. **Versions** までスクロールします。
5. アーティファクト バージョンの横にある **View** をクリックします。
6. **Version** タブ内で、タグ名にマウスを合わせます。
7. キャンセル ボタン (**X** アイコン) をクリックします。

## 既存のタグを検索する

W&B App UI を使用して、コレクションやアーティファクト バージョン内の既存のタグを検索します。

1. [W&B Registry App](https://wandb.ai/registry) に移動します。
2. レジストリ カードをクリックします。
3. 検索バー内で、タグの名前を入力します。

{{< img src="/images/registry/search_tags.gif" alt="タグベース検索" >}}


## 特定のタグを持つアーティファクト バージョンを検索する

W&B Python SDK を使用して、特定のタグを持つアーティファクト バージョンを検索します。

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