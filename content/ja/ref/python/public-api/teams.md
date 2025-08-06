---
title: チーム
object_type: public_apis_namespace
data_type_classification: module
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/apis/public/teams.py >}}




# <kbd>module</kbd> `wandb.apis.public`
W&B Public API で Teams やチームメンバーの管理を行うためのモジュールです。

このモジュールは、W&B の Teams とそのメンバー管理用のクラスを提供します。



**注意:**

> このモジュールは W&B Public API の一部であり、Teams やそのメンバーを管理するメソッドを提供します。Team の管理操作には適切な権限が必要です。



---

## <kbd>class</kbd> `Member`
チームのメンバーを表すクラスです。



**引数:**
 
 - `client` (`wandb.apis.internal.Api`):  使用するクライアントインスタンス
 - `team` (str):  このメンバーが所属するチーム名
 - `attrs` (dict):  メンバーの属性情報

### <kbd>method</kbd> `Member.__init__`

```python
__init__(client, team, attrs)
```








---

### <kbd>method</kbd> `Member.delete`

```python
delete()
```

チームからメンバーを削除します。



**戻り値:**
  削除が成功したかどうかのブール値


---

## <kbd>class</kbd> `Team`
W&B の Team を表すクラスです。

このクラスは、Team の作成、メンバー招待、サービスアカウント管理など、W&B Team の管理用メソッドを提供します。Attrs を継承してチーム属性の管理も行います。



**引数:**
 
 - `client` (`wandb.apis.public.Api`):  使用する API インスタンス
 - `name` (str):  チーム名
 - `attrs` (dict):  オプション。チーム属性の辞書



**注意:**

> Team の管理には適切な権限が必要です。

### <kbd>method</kbd> `Team.__init__`

```python
__init__(client, name, attrs=None)
```








---

### <kbd>classmethod</kbd> `Team.create`

```python
create(api, team, admin_username=None)
```

新しい Team を作成します。



**引数:**
 
 - `api`:  (`Api`) 使用する API インスタンス
 - `team`:  (str) チーム名
 - `admin_username`:  (str) オプション。チームの管理ユーザー名。デフォルトは現在のユーザー。



**戻り値:**
 `Team` オブジェクト

---

### <kbd>method</kbd> `Team.create_service_account`

```python
create_service_account(description)
```

チーム用のサービスアカウントを作成します。



**引数:**
 
 - `description`:  (str) このサービスアカウントの説明



**戻り値:**
 サービスアカウントの `Member` オブジェクト。失敗時は None

---

### <kbd>method</kbd> `Team.invite`

```python
invite(username_or_email, admin=False)
```

ユーザーを Team に招待します。



**引数:**
 
 - `username_or_email`:  (str) 招待したいユーザーのユーザー名またはメールアドレス
 - `admin`:  (bool) このユーザーを Team 管理者にするかどうか。デフォルトは `False`。



**戻り値:**
 成功時は `True`。既に招待済、またはユーザーが存在しない場合は `False`。

---