---
title: login()
object_type: python_sdk_actions
data_type_classification: function
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

デフォルトでは、認証情報はローカルにのみ保存され、W&B サーバーで認証されません。認証情報を検証したい場合は `verify=True` を指定してください。



**引数:**

- `anonymous`:  "must"、"allow"、"never" のいずれかを指定します。"must" の場合は常に匿名ユーザーとしてログインします。"allow" の場合、すでにログインしていない場合のみ匿名ユーザーが作成されます。"never" の場合は決して匿名ユーザーでログインしません。デフォルトは "never" です。
- `key`:  使用する APIキー です。
- `relogin`:  True の場合、APIキー の再入力を促します。
- `host`:  接続するホスト名です。
- `force`:  True の場合、強制的に再ログインを実施します。
- `timeout`:  ユーザー入力を待つ秒数です。
- `verify`:  W&B サーバーで認証情報を検証します。
- `referrer`:  URL ログインリクエストで利用するリファラ情報です。



**戻り値:**

- `bool`:  `key` が設定されている場合は True。



**例外:**

- `AuthenticationError`:  `api_key` の認証がサーバーで失敗した場合に発生します。
- `UsageError`:  `api_key` を設定できず、かつ tty が使用できない場合に発生します。
