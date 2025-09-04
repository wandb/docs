---
title: login
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/v0.21.3/wandb/sdk/wandb_login.py#L41-L89 >}}

Set up W&B login credentials.

```python
login(
    anonymous: Optional[Literal['must', 'allow', 'never']] = None,
    key: Optional[str] = None,
    relogin: Optional[bool] = None,
    host: Optional[str] = None,
    force: Optional[bool] = None,
    timeout: Optional[int] = None,
    verify: bool = (False),
    referrer: Optional[str] = None
) -> bool
```

By default, this will only store credentials locally without
verifying them with the W&B server. To verify credentials, pass
`verify=True`.

| Args |  |
| :--- | :--- |
|  `anonymous` |  Set to "must", "allow", or "never". If set to "must", always log a user in anonymously. If set to "allow", only create an anonymous user if the user isn't already logged in. If set to "never", never log a user anonymously. Default set to "never". Defaults to `None`. |
|  `key` |  The API key to use. |
|  `relogin` |  If true, will re-prompt for API key. |
|  `host` |  The host to connect to. |
|  `force` |  If true, will force a relogin. |
|  `timeout` |  Number of seconds to wait for user input. |
|  `verify` |  Verify the credentials with the W&B server. |
|  `referrer` |  The referrer to use in the URL login request. |

| Returns |  |
| :--- | :--- |
|  `bool` |  If `key` is configured. |

| Raises |  |
| :--- | :--- |
|  `AuthenticationError` |  If `api_key` fails verification with the server. |
|  `UsageError` |  If `api_key` cannot be configured and no tty. |
