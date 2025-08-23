---
title: アーティファクト
data_type_classification: module
menu:
  reference:
    identifier: ja-ref-python-public-api-artifacts
object_type: public_apis_namespace
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/apis/public/artifacts.py >}}




# <kbd>module</kbd> `wandb.apis.public`
W&B パブリックAPI：Artifact オブジェクト用

このモジュールは W&B Artifacts およびそのコレクションとやり取りするためのクラスを提供します。


## <kbd>class</kbd> `ArtifactTypes`
特定のプロジェクト内での Artifacts タイプのイテラブルなコレクションです。


## <kbd>class</kbd> `ArtifactType`
指定されたタイプにもとづくクエリに合致した Artifact オブジェクト。

**引数:**
 
 - `client`:  クエリに使用する W&B クライアント インスタンス 
 - `entity`:  プロジェクトの所有者（ユーザーまたはチーム） 
 - `project`:  Artifact タイプを検索したいプロジェクト名 
 - `type_name`:  Artifact タイプ名
 - `attrs`:  Artifact タイプ初期化時の属性マッピング（省略時は初期化時に W&B からロードされます）


### <kbd>property</kbd> ArtifactType.id

artifact タイプの一意な識別子

---

### <kbd>property</kbd> ArtifactType.name

artifact タイプの名前


---

### <kbd>method</kbd> `ArtifactType.collection`

```python
collection(name: 'str') → ArtifactCollection
```

指定した名前の Artifact コレクションを取得します。

**引数:**
 
 - `name` (str):  取得したい Artifact コレクションの名前

---

### <kbd>method</kbd> `ArtifactType.collections`

```python
collections(per_page: 'int' = 50) → ArtifactCollections
```

この artifact タイプに紐づくすべてのコレクションを取得します。

**引数:**
 
 - `per_page` (int):  1ページあたり取得する Artifact コレクション数（デフォルト: 50）

---


## <kbd>class</kbd> `ArtifactCollections`
プロジェクト内の特定タイプに属する Artifact コレクションの集合


**引数:**
 
 - `client`:  クエリに使用する W&B クライアント インスタンス
 - `entity`:  プロジェクトの所有者（ユーザーまたはチーム）
 - `project`:  Artifact コレクションを検索するプロジェクト名
 - `type_name`:  取得するコレクションの Artifact タイプ名
 - `per_page`:  1ページあたり取得する Artifact コレクション数（デフォルト: 50）

### <kbd>property</kbd> ArtifactCollections.length





---




## <kbd>class</kbd> `ArtifactCollection`
関連する Artifact をまとめるグループコレクション


**引数:**
 
 - `client`:  クエリに使用する W&B クライアント インスタンス
 - `entity`:  プロジェクトの所有者（ユーザーまたはチーム）
 - `project`:  Artifact コレクションを検索するプロジェクト名
 - `name`:  Artifact コレクション名
 - `type`:  Artifact コレクションのタイプ（例: "dataset", "model" など）
 - `organization`:  必要に応じて組織名
 - `attrs`:  初期化時に用いる属性マッピング（省略時は初期化時に W&B からロード）


### <kbd>property</kbd> ArtifactCollection.aliases

Artifact コレクションのエイリアス

---

### <kbd>property</kbd> ArtifactCollection.created_at

Artifact コレクションの作成日

---

### <kbd>property</kbd> ArtifactCollection.description

この Artifact コレクションの説明

---

### <kbd>property</kbd> ArtifactCollection.id

Artifact コレクションの一意な識別子

---

### <kbd>property</kbd> ArtifactCollection.name

Artifact コレクションの名前

---

### <kbd>property</kbd> ArtifactCollection.tags

Artifact コレクションに紐づくタグ群

---

### <kbd>property</kbd> ArtifactCollection.type

Artifact コレクションのタイプを返します。


---

### <kbd>method</kbd> `ArtifactCollection.artifacts`

```python
artifacts(per_page: 'int' = 50) → Artifacts
```

コレクション内のすべての Artifact を取得します

---

### <kbd>method</kbd> `ArtifactCollection.change_type`

```python
change_type(new_type: 'str') → None
```

非推奨：タイプ変更は `save` で直接行ってください

---

### <kbd>method</kbd> `ArtifactCollection.delete`

```python
delete() → None
```

この Artifact コレクション全体を削除します

---

### <kbd>method</kbd> `ArtifactCollection.is_sequence`

```python
is_sequence() → bool
```

この Artifact コレクションがシーケンスかどうかを返します

---


### <kbd>method</kbd> `ArtifactCollection.save`

```python
save() → None
```

Artifact コレクションへの変更を保存します


---

## <kbd>class</kbd> `Artifacts`
プロジェクトに紐づく Artifact バージョンのイテラブルなコレクション

特定条件でも検索できるようフィルタも渡せます

**引数:**
 
 - `client`:  クエリに使用する W&B クライアント インスタンス
 - `entity`:  プロジェクトの所有者（ユーザーまたはチーム）
 - `project`:  Artifact を検索するプロジェクト名
 - `collection_name`:  クエリする Artifact コレクション名
 - `type`:  検索する Artifact のタイプ（例: "dataset" や "model"）
 - `filters`:  クエリに適用するフィルタのマッピング（任意）
 - `order`:  結果の並び順指定用の文字列（任意）
 - `per_page`:  1ページあたり取得する Artifact バージョン数（デフォルト: 50）
 - `tags`:  タグで Artifact を絞り込みたい場合の文字列または文字列リスト（任意）

### <kbd>property</kbd> Artifacts.length





---



## <kbd>class</kbd> `RunArtifacts`
特定の run に紐づく Artifact のイテラブルなコレクション


### <kbd>property</kbd> RunArtifacts.length





---



## <kbd>class</kbd> `ArtifactFiles`
Artifact 内ファイル用のページネータ


### <kbd>property</kbd> ArtifactFiles.length





---


### <kbd>property</kbd> ArtifactFiles.path

Artifact のパスを返します


---