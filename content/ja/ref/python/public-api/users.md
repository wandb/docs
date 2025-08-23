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
W&B Public API でユーザーとAPIキーを管理するためのモジュールです。

このモジュールは、W&B ユーザーおよびその APIキーを管理するためのクラスを提供します。



**注:**

> このモジュールは W&B Public API の一部であり、ユーザーとその認証を管理するメソッドを提供します。一部の操作には管理者権限が必要です。



---

## <kbd>class</kbd> `User`
W&B ユーザーの認証と管理機能を持つクラス。

このクラスは W&B ユーザーの作成、APIキーの管理、チームメンバーシップへのアクセスなどのメソッドを提供します。ユーザー属性の管理には Attrs を継承しています。



**引数:**
 
 - `client`:  (`wandb.apis.internal.Api`) 利用するクライアントインスタンス
 - `attrs`:  (dict) ユーザー属性



**注:**

> 一部の操作には管理者権限が必要です

### <kbd>method</kbd> `User.__init__`

```python
__init__(client, attrs)
```
<!--
初期化メソッドです。
-->



---

### <kbd>property</kbd> User.api_keys

ユーザーに紐付いている APIキー名のリスト。



**戻り値:**
 
 - `list[str]`:  ユーザーに紐付いている APIキー名のリスト。ユーザーが APIキーを持っていない、もしくはAPIキーのデータが未ロードの場合は空リストを返します。

---

### <kbd>property</kbd> User.teams

ユーザーが所属しているチーム名のリスト。



**戻り値:**
 
 - `list` (list):  ユーザーが所属するチーム名のリスト。チームメンバーでないか、チームデータが未ロードの場合は空リストを返します。

---

### <kbd>property</kbd> User.user_api

ユーザーの認証情報を使った api のインスタンス。



---

### <kbd>classmethod</kbd> `User.create`

```python
create(api, email, admin=False)
```

新しいユーザーを作成します。



**引数:**
 
 - `api` (`Api`):  利用する api インスタンス
 - `email` (str):  チーム名
 - `admin` (bool):  このユーザーをグローバルインスタンス管理者にするかどうか



**戻り値:**
 `User` オブジェクト

---

### <kbd>method</kbd> `User.delete_api_key`

```python
delete_api_key(api_key)
```

ユーザーの APIキーを削除します。



**引数:**
 
 - `api_key` (str):  削除する APIキーの名前。これは `api_keys` プロパティが返すいずれかの名前である必要があります。



**戻り値:**
 成功なら True、失敗なら False のブール値



**例外:**
 api_key が見つからなかった場合は ValueError を送出します

---

### <kbd>method</kbd> `User.generate_api_key`

```python
generate_api_key(description=None)
```

新しい APIキーを生成します。



**引数:**
 
 - `description` (str, オプション):  新しい APIキーの説明。APIキーの用途を識別するために使用できます。



**戻り値:**
 新しい APIキー。失敗した場合は None
