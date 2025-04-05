---
title: login
object_type: api
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/wandb/sdk/wandb_login.py >}}




### <kbd>function</kbd> `login`

```python
login(
    anonymous: Optional[Literal['must', 'allow', 'never']] = None,
    key: Optional[str] = None,
    relogin: Optional[bool] = None,
    host: Optional[str] = None,
    force: Optional[bool] = None,
    timeout: Optional[int] = None,
    verify: bool = False
) → bool
```

Set up W&B login credentials. 

By default, this will only store credentials locally without verifying them with the W&B server. To verify credentials, pass `verify=True`. 



**Args:**
 
 - `anonymous`:  Set to `must`, `allow`, or `never`.  If set to `must`, always log a user in anonymously. If set to  "allow", only create an anonymous user if the user  isn't already logged in. If set to `never`, never log a  user anonymously. Default set to `never`. 
 - `key`:  The API key to use. 
 - `relogin`:  If true, will re-prompt for API key. 
 - `host`:  The host to connect to. 
 - `force`:  If true, will force a relogin. 
 - `timeout`:  Number of seconds to wait for user input. 
 - `verify`:  Verify the credentials with the W&B server. 



**Returns:**
 
 - `bool`:  if key is configured 



**Raises:**
 `AuthenticationError` if api_key fails verification with the server `UsageError` if api_key cannot be configured and no tty 
