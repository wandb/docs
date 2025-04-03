---
title: Organize versions with tags
description: タグを使用して、コレクションまたはコレクション内のアーティファクト のバージョンを整理します。タグの追加、削除、編集は、Python SDK
  または W&B App UI で行えます。
menu:
  default:
    identifier: ja-guides-core-registry-organize-with-tags
    parent: registry
weight: 7
---

レジストリ内のコレクションまたは Artifacts バージョンを整理するために、タグを作成して追加します。W&B App UIまたは W&B Python SDK を使用して、コレクションまたは Artifacts バージョンにタグを追加、変更、表示、または削除します。

{{% alert title="タグとエイリアスの使い分け" %}}
特定の Artifacts バージョンを一意に参照する必要がある場合は、エイリアスを使用します。たとえば、`artifact_name:alias` が常に単一の特定のバージョンを指すようにするために、'production' や 'latest' などのエイリアスを使用します。

グループ化や検索の柔軟性を高めたい場合は、タグを使用します。タグは、複数のバージョンまたはコレクションが同じラベルを共有でき、特定の識別子に1つのバージョンのみが関連付けられているという保証が不要な場合に最適です。
{{% /alert %}}

## コレクションにタグを追加する

W&B App UIまたは Python SDK を使用して、コレクションにタグを追加します。

{{< tabpane text=true >}}
{{% tab header="W&B App" %}}

W&B App UIを使用して、コレクションにタグを追加します。

1. W&B Registry（https://wandb.ai/registry）に移動します。
2. レジストリカードをクリックします。
3. コレクションの名前の横にある [**詳細を表示**] をクリックします。
4. コレクションカード内で、[**タグ**] フィールドの横にあるプラスアイコン（**+**）をクリックし、タグの名前を入力します。
5. キーボードの **Enter** キーを押します。

{{< img src="/images/registry/add_tag_collection.gif" alt="" >}}

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

`tags` 属性を再割り当てするか、変更することで、プログラムでタグを更新します。W&B は、インプレースミューテーションではなく、`tags` 属性を再割り当てすることを推奨しており、これは良い Python のプラクティスです。

たとえば、次のコードスニペットは、再割り当てでリストを更新する一般的な方法を示しています。簡潔にするために、[コレクションにタグを追加するセクション]({{< relref path="#add-a-tag-to-a-collection" lang="ja" >}})のコード例を続けます。

```python
collection.tags = [*collection.tags, "new-tag", "other-tag"]
collection.tags = collection.tags + ["new-tag", "other-tag"]

collection.tags = set(collection.tags) - set(tags_to_delete)
collection.tags = []  # deletes all tags
```

次のコードスニペットは、インプレースミューテーションを使用して、Artifacts バージョンに属するタグを更新する方法を示しています。

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

W&B App UIを使用して、コレクションに追加されたタグを表示します。

1. W&B Registry（https://wandb.ai/registry）に移動します。
2. レジストリカードをクリックします。
3. コレクションの名前の横にある [**詳細を表示**] をクリックします。

コレクションに1つ以上のタグがある場合、[**タグ**] フィールドの横にあるコレクションカード内でそれらのタグを表示できます。

{{< img src="/images/registry/tag_collection_selected.png" alt="" >}}

コレクションに追加されたタグは、そのコレクションの名前の横にも表示されます。

たとえば、上記の画像では、「tag1」というタグが「zoo-dataset-tensors」コレクションに追加されました。

{{< img src="/images/registry/tag_collection.png" alt="" >}}

## コレクションからタグを削除する

W&B App UIを使用して、コレクションからタグを削除します。

1. W&B Registry（https://wandb.ai/registry）に移動します。
2. レジストリカードをクリックします。
3. コレクションの名前の横にある [**詳細を表示**] をクリックします。
4. コレクションカード内で、削除するタグの名前にマウスを合わせます。
5. キャンセルボタン（**X** アイコン）をクリックします。

## Artifacts バージョンにタグを追加する

W&B App UIまたは Python SDK を使用して、コレクションにリンクされた Artifacts バージョンにタグを追加します。

{{< tabpane text=true >}}
{{% tab header="W&B App" %}}
1. W&B Registry（https://wandb.ai/registry）に移動します。
2. レジストリカードをクリックします。
3. タグを追加するコレクションの名前の横にある [**詳細を表示**] をクリックします。
4. [**バージョン**] までスクロールダウンします。
5. Artifacts バージョンの横にある [**表示**] をクリックします。
6. [**バージョン**] タブ内で、[**タグ**] フィールドの横にあるプラスアイコン（**+**）をクリックし、タグの名前を入力します。
7. キーボードの **Enter** キーを押します。

{{< img src="/images/registry/add_tag_linked_artifact_version.gif" alt="" >}}

{{% /tab %}}
{{% tab header="Python SDK" %}}
タグを追加または更新する Artifacts バージョンをフェッチします。Artifacts バージョンを取得したら、Artifacts オブジェクトの `tag` 属性にアクセスして、その Artifacts にタグを追加または変更できます。Artifacts の `tag` 属性に1つまたは複数のタグをリストとして渡します。

他の Artifacts と同様に、run を作成せずに W&B から Artifacts をフェッチするか、run を作成してその run 内で Artifacts をフェッチできます。いずれの場合も、Artifacts オブジェクトの `save` メソッドを呼び出して、W&B サーバー上の Artifacts を更新してください。

以下の適切なコードセルをコピーして貼り付け、Artifacts バージョンのタグを追加または変更します。`<>` の値を自分の値に置き換えます。

次のコードスニペットは、新しい run を作成せずに Artifacts をフェッチしてタグを追加する方法を示しています。
```python title="新しい run を作成せずに Artifacts バージョンにタグを追加する"
import wandb

ARTIFACT_TYPE = "<TYPE>"
ORG_NAME = "<org_name>"
REGISTRY_NAME = "<registry_name>"
COLLECTION_NAME = "<collection_name>"
VERSION = "<artifact_version>"

artifact_name = f"{ORG_NAME}/wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}:v{VERSION}"

artifact = wandb.Api().artifact(name = artifact_name, type = ARTIFACT_TYPE)
artifact.tags = ["tag2"] # リストで1つ以上のタグを提供します
artifact.save()
```

次のコードスニペットは、新しい run を作成して Artifacts をフェッチし、タグを追加する方法を示しています。

```python title="run 中に Artifacts バージョンにタグを追加する"
import wandb

ORG_NAME = "<org_name>"
REGISTRY_NAME = "<registry_name>"
COLLECTION_NAME = "<collection_name>"
VERSION = "<artifact_version>"

run = wandb.init(entity = "<entity>", project="<project>")

artifact_name = f"{ORG_NAME}/wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}:v{VERSION}"

artifact = run.use_artifact(artifact_or_name = artifact_name)
artifact.tags = ["tag2"] # リストで1つ以上のタグを提供します
artifact.save()
```

{{% /tab %}}
{{< /tabpane >}}

## Artifacts バージョンに属するタグを更新する

`tags` 属性を再割り当てするか、変更することで、プログラムでタグを更新します。W&B は、インプレースミューテーションではなく、`tags` 属性を再割り当てすることを推奨しており、これは良い Python のプラクティスです。

たとえば、次のコードスニペットは、再割り当てでリストを更新する一般的な方法を示しています。簡潔にするために、[Artifacts バージョンにタグを追加するセクション]({{< relref path="#add-a-tag-to-an-artifact-version" lang="ja" >}})のコード例を続けます。

```python
artifact.tags = [*artifact.tags, "new-tag", "other-tag"]
artifact.tags = artifact.tags + ["new-tag", "other-tag"]

artifact.tags = set(artifact.tags) - set(tags_to_delete)
artifact.tags = []  # deletes all tags
```

次のコードスニペットは、インプレースミューテーションを使用して、Artifacts バージョンに属するタグを更新する方法を示しています。

```python
artifact.tags += ["new-tag", "other-tag"]
artifact.tags.append("new-tag")

artifact.tags.extend(["new-tag", "other-tag"])
artifact.tags[:] = ["new-tag", "other-tag"]
artifact.tags.remove("existing-tag")
artifact.tags.pop()
artifact.tags.clear()
```

## Artifacts バージョンに属するタグを表示する

W&B App UIまたは Python SDK を使用して、レジストリにリンクされている Artifacts バージョンに属するタグを表示します。

{{< tabpane text=true >}}
{{% tab header="W&B App" %}}

1. W&B Registry（https://wandb.ai/registry）に移動します。
2. レジストリカードをクリックします。
3. タグを追加するコレクションの名前の横にある [**詳細を表示**] をクリックします。
4. [**バージョン**] セクションまでスクロールダウンします。

Artifacts バージョンに1つ以上のタグがある場合、[**タグ**] 列内でそれらのタグを表示できます。

{{< img src="/images/registry/tag_artifact_version.png" alt="" >}}

{{% /tab %}}
{{% tab header="Python SDK" %}}

タグを表示する Artifacts バージョンをフェッチします。Artifacts バージョンを取得したら、Artifacts オブジェクトの `tag` 属性を表示して、その Artifacts に属するタグを表示できます。

他の Artifacts と同様に、run を作成せずに W&B から Artifacts をフェッチするか、run を作成してその run 内で Artifacts をフェッチできます。

以下の適切なコードセルをコピーして貼り付け、Artifacts バージョンのタグを追加または変更します。`<>` の値を自分の値に置き換えます。

次のコードスニペットは、新しい run を作成せずに Artifacts バージョンのタグをフェッチして表示する方法を示しています。

```python title="新しい run を作成せずに Artifacts バージョンにタグを追加する"
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

次のコードスニペットは、新しい run を作成して Artifacts バージョンのタグをフェッチして表示する方法を示しています。

```python title="run 中に Artifacts バージョンにタグを追加する"
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

## Artifacts バージョンからタグを削除する

1. W&B Registry（https://wandb.ai/registry）に移動します。
2. レジストリカードをクリックします。
3. タグを追加するコレクションの名前の横にある [**詳細を表示**] をクリックします。
4. [**バージョン**] までスクロールダウンします。
5. Artifacts バージョンの横にある [**表示**] をクリックします。
6. [**バージョン**] タブ内で、タグの名前にマウスを合わせます。
7. キャンセルボタン（**X** アイコン）をクリックします。

## 既存のタグを検索する

W&B App UIを使用して、コレクションおよび Artifacts バージョンで既存のタグを検索します。

1. W&B Registry（https://wandb.ai/registry）に移動します。
2. レジストリカードをクリックします。
3. 検索バーにタグの名前を入力します。

{{< img src="/images/registry/search_tags.gif" alt="" >}}

## 特定のタグを持つ Artifacts バージョンを検索する

W&B Python SDK を使用して、タグのセットを持つ Artifacts バージョンを検索します。

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
