---
title: login()
data_type_classification: function
menu:
  reference:
    identifier: ja-ref-python-sdk-functions-login
object_type: python_sdk_actions
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_login.py >}}




### <kbd>function</kbd> `login`

```python
login(
    anonymous: Optional[Literal['allow', 'must', 'never']] = None,
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

デフォルトでは、認証情報はローカルにのみ保存され、W&B サーバーでの検証は行いません。認証情報を検証したい場合は、`verify=True` を指定してください。



**引数:**
 
 - `anonymous`:  "must"、"allow"、"never" のいずれかを設定します。"must" の場合、常に匿名ユーザーでログインします。"allow" の場合、すでにユーザーがログインしていない場合のみ匿名ユーザーを作成します。"never" の場合は匿名ユーザーでのログインを行いません。デフォルトは "never" です。
 - `key`:  使用する APIキー。
 - `relogin`:  True にすると再度 APIキー の入力を求めます。
 - `host`:  接続先のホスト。
 - `force`:  True の場合、再ログインを強制します。
 - `timeout`:  ユーザー入力を待つ秒数。
 - `verify`:  W&B サーバーで認証情報を検証します。
 - `referrer`:  URLログインリクエストで利用するリファラー。



**戻り値:**
 
 - `bool`:  `key` が設定されていれば True を返します。



**例外:**
 
 - `AuthenticationError`:  `api_key` のサーバーでの検証に失敗した場合に発生します。
 - `UsageError`:  `api_key` を設定できず、tty も利用できない場合に発生します。
