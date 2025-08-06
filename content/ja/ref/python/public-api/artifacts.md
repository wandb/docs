---
title: アーティファクト
object_type: public_apis_namespace
data_type_classification: module
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/apis/public/artifacts.py >}}




# <kbd>module</kbd> `wandb.apis.public`
W&B の Artifact オブジェクト向けパブリック API。

このモジュールは、W&B Artifacts とそのコレクションを操作するためのクラスを提供します。


## <kbd>class</kbd> `ArtifactTypes`
特定のプロジェクトに対する artifact type のイテラブルコレクション。


## <kbd>class</kbd> `ArtifactType`
指定された type に基づくクエリを満たす Artifact オブジェクト。



**Args:**
 
 - `client`:  W&B へのクエリに使用するクライアントインスタンス。
 - `entity`:  プロジェクトのオーナーとなるエンティティ（ユーザーまたはチーム）。
 - `project`:  artifact type を問い合わせるプロジェクト名。
 - `type_name`:  artifact type の名前。
 - `attrs`:  オプションで、artifact type の初期化に使う属性のマッピング。指定しない場合、初期化時に W&B から属性を読み込みます。


### <kbd>property</kbd> ArtifactType.id

artifact type のユニークな識別子。

---

### <kbd>property</kbd> ArtifactType.name

artifact type の名前。



---

### <kbd>method</kbd> `ArtifactType.collection`

```python
collection(name: 'str') → ArtifactCollection
```

指定した名前の artifact コレクションを取得します。



**Args:**
 
 - `name` (str):  取得したい artifact コレクションの名前。

---

### <kbd>method</kbd> `ArtifactType.collections`

```python
collections(per_page: 'int' = 50) → ArtifactCollections
```

この artifact type に紐づくすべての artifact コレクションを取得します。



**Args:**
 
 - `per_page` (int):  1ページあたり取得する artifact コレクション数。デフォルトは50。

---


## <kbd>class</kbd> `ArtifactCollections`
プロジェクト内の特定タイプの artifact コレクション群。



**Args:**
 
 - `client`:  W&B へのクエリに使用するクライアントインスタンス。
 - `entity`:  プロジェクトのオーナーとなるエンティティ（ユーザーまたはチーム）。
 - `project`:  artifact コレクションを問い合わせるプロジェクト名。
 - `type_name`:  コレクションを取得する対象となる artifact type の名前。
 - `per_page`:  1ページあたりに取得する artifact コレクション数。デフォルトは50。


### <kbd>property</kbd> ArtifactCollections.length





---




## <kbd>class</kbd> `ArtifactCollection`
関連する複数の Artifacts をまとめた artifact コレクション。



**Args:**
 
 - `client`:  W&B へのクエリに使用するクライアントインスタンス。
 - `entity`:  プロジェクトのオーナーとなるエンティティ（ユーザーまたはチーム）。
 - `project`:  artifact コレクションを問い合わせるプロジェクト名。
 - `name`:  artifact コレクションの名前。
 - `type`:  artifact コレクションのタイプ（例："dataset", "model" など）。
 - `organization`:  必要に応じて組織名を指定可能。
 - `attrs`:  オプションで、artifact コレクションの初期化属性マッピング。指定がない場合、初期化時に W&B から属性が読み込まれます。


### <kbd>property</kbd> ArtifactCollection.aliases

artifact コレクションのエイリアス。

---

### <kbd>property</kbd> ArtifactCollection.created_at

artifact コレクションの作成日時。

---

### <kbd>property</kbd> ArtifactCollection.description

artifact コレクションの説明。

---

### <kbd>property</kbd> ArtifactCollection.id

artifact コレクションのユニークな識別子。

---

### <kbd>property</kbd> ArtifactCollection.name

artifact コレクションの名前。

---

### <kbd>property</kbd> ArtifactCollection.tags

artifact コレクションに紐づくタグ。

---

### <kbd>property</kbd> ArtifactCollection.type

artifact コレクションのタイプを返します。



---

### <kbd>method</kbd> `ArtifactCollection.artifacts`

```python
artifacts(per_page: 'int' = 50) → Artifacts
```

コレクション内のすべての Artifacts を取得します。

---

### <kbd>method</kbd> `ArtifactCollection.change_type`

```python
change_type(new_type: 'str') → None
```

非推奨：type の変更は `save` を直接使用してください。

---

### <kbd>method</kbd> `ArtifactCollection.delete`

```python
delete() → None
```

artifact コレクション全体を削除します。

---

### <kbd>method</kbd> `ArtifactCollection.is_sequence`

```python
is_sequence() → bool
```

artifact コレクションがシーケンスかどうかを返します。

---


### <kbd>method</kbd> `ArtifactCollection.save`

```python
save() → None
```

artifact コレクションの変更内容を保存します。


---

## <kbd>class</kbd> `Artifacts`
プロジェクトに紐づく artifact バージョンのイテラブルコレクション。

オプションでフィルタを指定して、特定の条件で結果を絞り込むことができます。



**Args:**
 
 - `client`:  W&B へのクエリに使用するクライアントインスタンス。
 - `entity`:  プロジェクトのオーナーとなるエンティティ（ユーザーまたはチーム）。
 - `project`:  artifact を問い合わせるプロジェクト名。
 - `collection_name`:  問い合わせる artifact コレクションの名前。
 - `type`:  問い合わせる artifact のタイプ（例："dataset" や "model" など）。
 - `filters`:  クエリに適用するフィルタのマッピング（オプション）。
 - `order`:  結果の並び順を指定する文字列（オプション）。
 - `per_page`:  1ページあたりの artifact バージョン取得数。デフォルトは50。
 - `tags`:  フィルタする際のタグまたはタグのリスト（オプション）。


### <kbd>property</kbd> Artifacts.length





---



## <kbd>class</kbd> `RunArtifacts`
特定の run に紐づく artifact のイテラブルコレクション。


### <kbd>property</kbd> RunArtifacts.length





---



## <kbd>class</kbd> `ArtifactFiles`
artifact 内のファイル用ページネーター。


### <kbd>property</kbd> ArtifactFiles.length





---


### <kbd>property</kbd> ArtifactFiles.path

artifact のパスを返します。



---