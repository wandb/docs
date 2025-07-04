---
title: login
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/f1e324a66f6d9fd4ab7b43b66d9e832fa5e49b15/wandb/sdk/wandb_login.py#L41-L88 >}}

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
|  `anonymous` |  (string, optional) Can be "must", "allow", or "never". If set to "must", always log a user in anonymously. If set to "allow", only create an anonymous user if the user isn't already logged in. If set to "never", never log a user anonymously. Default set to "never". |
|  `key` |  (string, optional) The API key to use. |
|  `relogin` |  (bool, optional) If true, will re-prompt for API key. |
|  `host` |  (string, optional) The host to connect to. |
|  `force` |  (bool, optional) If true, will force a relogin. |
|  `timeout` |  (int, optional) Number of seconds to wait for user input. |
|  `verify` |  (bool) Verify the credentials with the W&B server. |
|  `referrer` |  (string, optional) The referrer to use in the URL login request. |

| Returns |  |
| :--- | :--- |
|  `bool` |  if key is configured |

| Raises |  |
| :--- | :--- |
|  AuthenticationError - if api_key fails verification with the server UsageError - if api_key cannot be configured and no tty |
