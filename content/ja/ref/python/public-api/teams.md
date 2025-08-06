---
title: チーム
data_type_classification: module
menu:
  reference:
    identifier: ja-ref-python-public-api-teams
object_type: public_apis_namespace
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/apis/public/teams.py >}}




# <kbd>module</kbd> `wandb.apis.public`
W&B Public API で Teams およびチームメンバーを管理します。

このモジュールは、W&B の Teams およびそのメンバー管理用のクラスを提供します。



**注:**

> このモジュールは W&B Public API の一部であり、Teams とそのメンバーを管理するメソッドを提供します。チーム管理には適切な権限が必要です。



---

## <kbd>class</kbd> `Member`
Team のメンバーを表すクラスです。



**引数:**
 
 - `client` (`wandb.apis.internal.Api`):  使用するクライアントインスタンス
 - `team` (str):  このメンバーが所属するチーム名
 - `attrs` (dict):  メンバー属性

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



**返り値:**
  成功の場合は True を返します


---

## <kbd>class</kbd> `Team`
W&B Team を表すクラスです。

このクラスは、Teams の作成、メンバー招待、サービスアカウントの管理など、W&B Teams の管理メソッドを提供します。チーム属性を扱うために `Attrs` から継承しています。



**引数:**
 
 - `client` (`wandb.apis.public.Api`):  使用する API インスタンス
 - `name` (str):  チーム名
 - `attrs` (dict):  オプションのチーム属性の辞書



**注:**

> チームの管理には適切な権限が必要です。

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
 - `admin_username`:  (str) チーム管理者のユーザー名（任意）。省略時は現在のユーザーになります。



**返り値:**
 `Team` オブジェクト

---

### <kbd>method</kbd> `Team.create_service_account`

```python
create_service_account(description)
```

Team 用のサービスアカウントを作成します。



**引数:**
 
 - `description`:  (str) このサービスアカウントの説明



**返り値:**
 サービスアカウントの `Member` オブジェクト。失敗時は None

---

### <kbd>method</kbd> `Team.invite`

```python
invite(username_or_email, admin=False)
```

ユーザーを Team に招待します。



**引数:**
 
 - `username_or_email`:  (str) 招待したいユーザーのユーザー名またはメールアドレス
 - `admin`:  (bool) このユーザーをチーム管理者にするか。デフォルトは `False`



**返り値:**
 成功時は `True`。既に招待済み、または存在しない場合は `False`

---