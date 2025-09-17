---
title: teams
data_type_classification: module
menu:
  reference:
    identifier: ja-ref-python-public-api-teams
object_type: public_apis_namespace
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/apis/public/teams.py >}}




# <kbd>module</kbd> `wandb.apis.public`
Teams と チームメンバーを管理するための W&B Public API。 

このモジュールは、W&B の Teams とそのメンバーを管理するためのクラスを提供します。 



**注意:**

> このモジュールは W&B Public API の一部であり、Teams とそのメンバーを管理するためのメソッドを提供します。Team の管理操作には適切な権限が必要です。 



---

## <kbd>class</kbd> `Member`
Team のメンバー。 



**引数:**
 
 - `client` (`wandb.apis.internal.Api`):  使用するクライアント インスタンス 
 - `team` (str):  このメンバーが所属する Team の名前 
 - `attrs` (dict):  メンバーの属性 

### <kbd>method</kbd> `Member.__init__`

```python
__init__(client, team, attrs)
```








---

### <kbd>method</kbd> `Member.delete`

```python
delete()
```

Team からメンバーを削除します。 



**戻り値:**
  成功したかどうかを示すブール値 


---

## <kbd>class</kbd> `Team`
W&B の Team を表すクラス。 

このクラスは、Teams の作成、メンバーの招待、サービス アカウントの管理など、W&B の Teams を管理するためのメソッドを提供します。Team の属性を扱うために Attrs を継承しています。 



**引数:**
 
 - `client` (`wandb.apis.public.Api`):  使用する API インスタンス 
 - `name` (str):  Team の名前 
 - `attrs` (dict):  省略可能な Team 属性の辞書 



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
 - `team`:  (str) Team の名前 
 - `admin_username`:  (str) Team の管理者ユーザーのユーザー名 (任意)。既定は現在のユーザーです。 



**戻り値:**
 `Team` オブジェクト 

---

### <kbd>method</kbd> `Team.create_service_account`

```python
create_service_account(description)
```

Team 用のサービス アカウントを作成します。 



**引数:**
 
 - `description`:  (str) このサービス アカウントの説明 



**戻り値:**
 サービス アカウントの `Member` オブジェクト。失敗した場合は None。 

---

### <kbd>method</kbd> `Team.invite`

```python
invite(username_or_email, admin=False)
```

ユーザーを Team に招待します。 



**引数:**
 
 - `username_or_email`:  (str) 招待したいユーザーのユーザー名またはメールアドレス。 
 - `admin`:  (bool) このユーザーを Team の管理者にするかどうか。既定は `False`。 



**戻り値:**
 成功した場合は `True`、すでに招待済み、または存在しないユーザーの場合は `False`。 

---