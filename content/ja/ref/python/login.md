---
title: login
menu:
  reference:
    identifier: ja-ref-python-login
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_login.py#L40-L84 >}}

W&B のログイン認証情報を設定します。

```python
login(
    anonymous: Optional[Literal['must', 'allow', 'never']] = None,
    key: Optional[str] = None,
    relogin: Optional[bool] = None,
    host: Optional[str] = None,
    force: Optional[bool] = None,
    timeout: Optional[int] = None,
    verify: bool = (False)
) -> bool
```

デフォルトでは、W&B サーバー で検証せずに、認証情報をローカルにのみ保存します。認証情報を検証するには、`verify=True` を渡します。

| Args |  |
| :--- | :--- |
|  `anonymous` |  (文字列、オプション) "must"、"allow"、または "never" を指定できます。"must" に設定すると、常に匿名で ユーザー をログインさせます。"allow" に設定すると、ユーザー がまだログインしていない場合にのみ、匿名 ユーザー を作成します。"never" に設定すると、匿名 ユーザー を決してログインさせません。デフォルトは "never" に設定されています。 |
|  `key` |  (文字列、オプション) 使用する APIキー 。 |
|  `relogin` |  (bool、オプション) Trueの場合、 APIキー の入力を再度求めます。 |
|  `host` |  (文字列、オプション) 接続先のホスト。 |
|  `force` |  (bool、オプション) Trueの場合、強制的に再ログインします。 |
|  `timeout` |  (int、オプション) ユーザー 入力を待つ秒数。 |
|  `verify` |  (bool) W&B サーバー で認証情報を検証します。 |

| Returns |  |
| :--- | :--- |
|  `bool` |  APIキー が設定されている場合 |

| Raises |  |
| :--- | :--- |
|  AuthenticationError - サーバー で api_key の検証に失敗した場合 UsageError - api_key を設定できず、tty がない場合 |
