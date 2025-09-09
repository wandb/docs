---
title: login()
data_type_classification: function
menu:
  reference:
    identifier: ja-ref-python-sdk-functions-login
object_type: python_sdk_actions
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_login.py >}}




### <kbd>関数</kbd> `login`

```python
login(
    anonymous: Optional[Literal['must', 'allow', 'never']] = None,
    key: Optional[str] = None,
    relogin: Optional[bool] = None,
    host: Optional[str] = None,
    force: Optional[bool] = None,
    timeout: Optional[int] = None,
    verify: bool = False,
    referrer: Optional[str] = None
) → bool
```

W&B のログイン認証情報を設定します。

既定では、W&B サーバーでの検証を行わずに認証情報をローカルにのみ保存します。認証情報を検証するには、`verify=True` を渡してください。



**引数:**
 
 - `anonymous`: "must"、"allow"、"never" のいずれかを指定します。 "must" の場合は常に匿名でユーザーをログインさせます。 "allow" の場合は、ユーザーが未ログインのときのみ匿名ユーザーを作成します。 "never" の場合は、匿名でユーザーをログインさせません。デフォルトは "never"。既定値は `None`。 
 - `key`: 使用する APIキー。 
 - `relogin`: true の場合、APIキー の入力を再度促します。 
 - `host`: 接続先のホスト。 
 - `force`: true の場合、再ログインを強制します。 
 - `timeout`: ユーザー入力を待機する秒数。 
 - `verify`: W&B サーバーで認証情報を検証します。 
 - `referrer`: URL ログインリクエストで使用するリファラ。 



**戻り値:**
 
 - `bool`: `key` が設定されていれば true。 



**例外:**
 
 - `AuthenticationError`: `api_key` のサーバーでの検証に失敗した場合。 
 - `UsageError`: `api_key` を設定できず、かつ tty がない場合。