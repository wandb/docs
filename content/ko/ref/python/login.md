---
title: login
menu:
  reference:
    identifier: ko-ref-python-login
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_login.py#L40-L84 >}}

W&B 로그인 자격 증명을 설정합니다.

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

기본적으로, 이는 W&B 서버로 검증하지 않고 로컬에만 자격 증명을 저장합니다. 자격 증명을 확인하려면 `verify=True`를 전달하십시오.

| Args |  |
| :--- | :--- |
|  `anonymous` |  (string, optional) "must", "allow" 또는 "never"일 수 있습니다. "must"로 설정하면 항상 익명으로 사용자를 로그인합니다. "allow"로 설정하면 사용자가 아직 로그인하지 않은 경우에만 익명 사용자를 만듭니다. "never"로 설정하면 익명으로 사용자를 로그인하지 않습니다. 기본값은 "never"로 설정됩니다. |
|  `key` |  (string, optional) 사용할 API 키입니다. |
|  `relogin` |  (bool, optional) true인 경우 API 키를 다시 묻습니다. |
|  `host` |  (string, optional) 연결할 호스트입니다. |
|  `force` |  (bool, optional) true인 경우 강제로 다시 로그인합니다. |
|  `timeout` |  (int, optional) 사용자 입력을 기다리는 시간(초)입니다. |
|  `verify` |  (bool) W&B 서버로 자격 증명을 확인합니다. |

| Returns |  |
| :--- | :--- |
|  `bool` |  키가 구성된 경우 |

| Raises |  |
| :--- | :--- |
|  AuthenticationError - API 키가 서버와의 확인에 실패한 경우 UsageError - API 키를 구성할 수 없고 tty가 없는 경우 |
