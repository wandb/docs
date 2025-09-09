---
title: アーティファクト
data_type_classification: module
menu:
  reference:
    identifier: ja-ref-python-public-api-artifacts
object_type: public_apis_namespace
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/apis/public/artifacts.py >}}




# <kbd>モジュール</kbd> `wandb.apis.public`
W&B の Artifact オブジェクト向け Public API。 

このモジュールは、W&B の Artifacts およびそのコレクションを操作するためのクラスを提供します。 


## <kbd>クラス</kbd> `ArtifactTypes`
特定の project 向け `ArtifactType` オブジェクトの遅延イテレーター。 


## <kbd>クラス</kbd> `ArtifactType`
指定されたタイプに基づくクエリを満たす Artifact オブジェクト。 



**引数:**
 
 - `client`:  W&B にクエリする際に使用するクライアントインスタンス。 
 - `entity`:  project の所有者である entity（user または team）。 
 - `project`:  Artifact Type を検索する対象の project 名。 
 - `type_name`:  Artifact Type の名前。 
 - `attrs`:  Artifact Type を初期化するための属性マッピング（任意）。指定しない場合、初期化時に W&B から属性を読み込みます。 


### <kbd>プロパティ</kbd> ArtifactType.id

Artifact Type の一意の識別子。 

---

### <kbd>プロパティ</kbd> ArtifactType.name

Artifact Type の名前。 



---

### <kbd>メソッド</kbd> `ArtifactType.collection`

```python
collection(name: 'str') → ArtifactCollection
```

指定した名前の artifact コレクションを取得します。 



**引数:**
 
 - `name` (str):  取得する artifact コレクション名。 

---

### <kbd>メソッド</kbd> `ArtifactType.collections`

```python
collections(per_page: 'int' = 50) → ArtifactCollections
```

この Artifact Type に関連付けられたすべての artifact コレクションを取得します。 



**引数:**
 
 - `per_page` (int):  1 ページあたりにフェッチする artifact コレクション数。デフォルトは 50。 

---


## <kbd>クラス</kbd> `ArtifactCollections`
project 内の特定タイプの artifact コレクション。 



**引数:**
 
 - `client`:  W&B にクエリする際に使用するクライアントインスタンス。 
 - `entity`:  project の所有者である entity（user または team）。 
 - `project`:  artifact コレクションを検索する対象の project 名。 
 - `type_name`:  コレクションを取得する対象の Artifact Type 名。 
 - `per_page`:  1 ページあたりにフェッチする artifact コレクション数。デフォルトは 50。 


### <kbd>プロパティ</kbd> ArtifactCollections.length





---




## <kbd>クラス</kbd> `ArtifactCollection`
関連する Artifacts のグループを表す artifact コレクション。 



**引数:**
 
 - `client`:  W&B にクエリする際に使用するクライアントインスタンス。 
 - `entity`:  project の所有者である entity（user または team）。 
 - `project`:  artifact コレクションを検索する対象の project 名。 
 - `name`:  artifact コレクションの名前。 
 - `type`:  artifact コレクションのタイプ（例: "dataset", "model"）。 
 - `organization`:  該当する場合の組織名（任意）。 
 - `attrs`:  artifact コレクションを初期化するための属性マッピング（任意）。指定しない場合、初期化時に W&B から属性を読み込みます。 


### <kbd>プロパティ</kbd> ArtifactCollection.aliases

Artifact Collection のエイリアス。 

---

### <kbd>プロパティ</kbd> ArtifactCollection.created_at

artifact コレクションの作成日時。 

---

### <kbd>プロパティ</kbd> ArtifactCollection.description

artifact コレクションの説明。 

---

### <kbd>プロパティ</kbd> ArtifactCollection.id

artifact コレクションの一意の識別子。 

---

### <kbd>プロパティ</kbd> ArtifactCollection.name

artifact コレクションの名前。 

---

### <kbd>プロパティ</kbd> ArtifactCollection.tags

artifact コレクションに関連付けられたタグ。 

---

### <kbd>プロパティ</kbd> ArtifactCollection.type

artifact コレクションのタイプを返します。 



---

### <kbd>メソッド</kbd> `ArtifactCollection.artifacts`

```python
artifacts(per_page: 'int' = 50) → Artifacts
```

コレクション内のすべての Artifacts を取得します。 

---

### <kbd>メソッド</kbd> `ArtifactCollection.change_type`

```python
change_type(new_type: 'str') → None
```

非推奨。代わりに `save` でタイプを直接変更してください。 

---

### <kbd>メソッド</kbd> `ArtifactCollection.delete`

```python
delete() → None
```

artifact コレクション全体を削除します。 

---

### <kbd>メソッド</kbd> `ArtifactCollection.is_sequence`

```python
is_sequence() → bool
```

artifact コレクションがシーケンスかどうかを返します。 

---


### <kbd>メソッド</kbd> `ArtifactCollection.save`

```python
save() → None
```

artifact コレクションに加えた変更を永続化します。 


---

## <kbd>クラス</kbd> `Artifacts`
project に関連付けられた artifact バージョンのイテラブルなコレクション。 

任意でフィルターを指定して、特定の条件に基づいて結果を絞り込めます。 



**引数:**
 
 - `client`:  W&B にクエリする際に使用するクライアントインスタンス。 
 - `entity`:  project の所有者である entity（user または team）。 
 - `project`:  Artifacts を検索する対象の project 名。 
 - `collection_name`:  クエリ対象の artifact コレクション名。 
 - `type`:  クエリする Artifacts のタイプ。一般的な例としては "dataset" や "model" があります。 
 - `filters`:  クエリに適用するフィルターのマッピング（任意）。 
 - `order`:  結果の並び順を指定する文字列（任意）。 
 - `per_page`:  1 ページあたりにフェッチする artifact バージョン数。デフォルトは 50。 
 - `tags`:  タグで Artifacts をフィルタリングするための文字列または文字列のリスト（任意）。 


### <kbd>プロパティ</kbd> Artifacts.length





---



## <kbd>クラス</kbd> `RunArtifacts`
特定の run に関連付けられた Artifacts のイテラブルなコレクション。 


### <kbd>プロパティ</kbd> RunArtifacts.length





---



## <kbd>クラス</kbd> `ArtifactFiles`
Artifact 内のファイルのページネーター。 


### <kbd>プロパティ</kbd> ArtifactFiles.length





---


### <kbd>プロパティ</kbd> ArtifactFiles.path

Artifact のパスを返します。 



---