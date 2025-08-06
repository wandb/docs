---
title: ユーザー
object_type: public_apis_namespace
data_type_classification: module
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/apis/public/users.py >}}




# <kbd>module</kbd> `wandb.apis.public`
W&B Public API でユーザーや APIキー を管理するためのモジュールです。

このモジュールは、W&B ユーザーとその APIキー を管理するためのクラスを提供します。



**注意:**

> このモジュールは W&B Public API の一部であり、ユーザーおよびその認証の管理メソッドを提供します。一部の操作には管理者権限が必要です。



---

## <kbd>class</kbd> `User`
W&Bユーザーを認証と管理の機能付きで表すクラスです。

このクラスは W&B ユーザーの管理（ユーザー作成、API キーの管理、チーム メンバーシップへのアクセスなど）を行うためのメソッドを提供します。ユーザー属性の管理には Attrs から継承しています。



**引数:**
 
 - `client`:  (`wandb.apis.internal.Api`) 利用するクライアントインスタンス
 - `attrs`:  (dict) ユーザーの属性



**注意:**

> 一部の操作には管理者権限が必要です

### <kbd>method</kbd> `User.__init__`

```python
__init__(client, attrs)
```
（ユーザーの初期化処理を行うコンストラクタです）





---

### <kbd>property</kbd> User.api_keys

ユーザーに紐づいている APIキー の名前一覧。



**戻り値:**
 
 - `list[str]`:  ユーザーに関連する APIキー の名前。APIキーがない場合や、APIキーのデータがまだ読み込まれていない場合は空リスト。 

---

### <kbd>property</kbd> User.teams

ユーザーが所属しているチーム名の一覧。



**戻り値:**
 
 - `list` (list):  ユーザーが所属しているチームの名前。チームメンバーシップが無い場合、またはチームデータがまだ読み込まれていない場合は空リスト。 

---

### <kbd>property</kbd> User.user_api

ユーザーの認証情報を使った api インスタンス。



---

### <kbd>classmethod</kbd> `User.create`

```python
create(api, email, admin=False)
```

新しいユーザーを作成します。



**引数:**
 
 - `api` (`Api`):  利用する api インスタンス
 - `email` (str):  チームの名前
 - `admin` (bool):  このユーザーをグローバル管理者にする場合はTrue



**戻り値:**
 `User` オブジェクト

---

### <kbd>method</kbd> `User.delete_api_key`

```python
delete_api_key(api_key)
```

ユーザーの APIキー を削除します。



**引数:**
 
 - `api_key` (str):  削除したい APIキー の名前。これは `api_keys` プロパティで返される名前のいずれかでなければなりません。



**戻り値:**
 成功時は True、失敗時は False



**例外:**
 api_key が見つからなかった場合、ValueError を発生させます

---

### <kbd>method</kbd> `User.generate_api_key`

```python
generate_api_key(description=None)
```

新しい APIキー を生成します。



**引数:**
 
 - `description` (str, オプション):  新しい APIキー の用途説明や識別のための説明文を指定できます。



**戻り値:**
 新しい APIキー。失敗時は None になります