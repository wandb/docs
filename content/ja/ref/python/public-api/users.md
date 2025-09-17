---
title: ユーザー
data_type_classification: module
menu:
  reference:
    identifier: ja-ref-python-public-api-users
object_type: public_apis_namespace
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/apis/public/users.py >}}




# <kbd>module</kbd> `wandb.apis.public`
ユーザーと APIキー を管理するための W&B Public API。 

このモジュールは、W&B のユーザーおよびその APIキー を管理するためのクラスを提供します。 



**Note:**

> このモジュールは W&B Public API の一部で、ユーザーとその認証を管理するメソッドを提供します。操作によっては管理者権限が必要です。 



---

## <kbd>class</kbd> `User`
認証および管理機能を備えた W&B のユーザーを表すクラス。 

ユーザーの作成、APIキー の管理、チーム メンバーシップへのアクセスなど、W&B ユーザーを管理するためのメソッドを提供します。ユーザーの属性を扱うために Attrs を継承しています。 



**Args:**
 
 - `client`:  (`wandb.apis.internal.Api`) 使用するクライアント インスタンス 
 - `attrs`:  (dict) ユーザー属性 



**Note:**

> 一部の操作には管理者権限が必要です 

### <kbd>method</kbd> `User.__init__`

```python
__init__(client, attrs)
```






---

### <kbd>property</kbd> User.api_keys

ユーザーに関連付けられた APIキー の名前一覧。 



**Returns:**
 
 - `list[str]`:  ユーザーに関連付けられた APIキー の名前。ユーザーに APIキー がない、または APIキー のデータが読み込まれていない場合は空リスト。 

---

### <kbd>property</kbd> User.teams

ユーザーが所属するチーム名の一覧。 



**Returns:**
 
 - `list` (list):  ユーザーが所属するチーム名。ユーザーにチームのメンバーシップがない、またはチームのデータが読み込まれていない場合は空リスト。 

---

### <kbd>property</kbd> User.user_api

このユーザーの認証情報を用いる API インスタンス。 



---

### <kbd>classmethod</kbd> `User.create`

```python
create(api, email, admin=False)
```

新しいユーザーを作成します。 



**Args:**
 
 - `api` (`Api`):  使用する API インスタンス 
 - `email` (str):  チームの名前 
 - `admin` (bool):  このユーザーをグローバル インスタンスの管理者にするかどうか 



**Returns:**
 `User` オブジェクト 

---

### <kbd>method</kbd> `User.delete_api_key`

```python
delete_api_key(api_key)
```

ユーザーの APIキー を削除します。 



**Args:**
 
 - `api_key` (str):  削除する APIキー の名前。これは `api_keys` プロパティ が返す名前のいずれかである必要があります。 



**Returns:**
 成功を示すブール値 



**Raises:**
 api_key が見つからない場合は ValueError 

---

### <kbd>method</kbd> `User.generate_api_key`

```python
generate_api_key(description=None)
```

新しい APIキー を生成します。 



**Args:**
 
 - `description` (str, optional):  新しい APIキー の説明。APIキー の用途の識別に使用できます。 



**Returns:**
 新しい APIキー。失敗した場合は None。
```