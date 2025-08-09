---
title: 'login()

  '
data_type_classification: function
menu:
  reference:
    identifier: ko-ref-python-sdk-functions-login
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

W&B 로그인 자격 정보를 설정합니다.

기본적으로, 이 함수는 자격 정보를 W&B 서버에 검증하지 않고 로컬에만 저장합니다. 자격 정보를 검증하려면 `verify=True`를 전달하세요.



**매개변수(Args):**
 
 - `anonymous`:  "must", "allow", "never" 중 하나로 설정합니다. "must"로 설정하면 항상 익명으로 사용자를 로그인합니다. "allow"로 설정하면, 사용자가 이미 로그인되어 있지 않은 경우에만 익명 사용자를 생성합니다. "never"로 설정하면 익명으로 사용자를 로그인하지 않습니다. 기본값은 "never"입니다.
 - `key`:  사용할 API 키입니다.
 - `relogin`:  True로 설정하면 API 키를 다시 입력받습니다.
 - `host`:  연결할 호스트입니다.
 - `force`:  True로 설정하면 로그인을 강제로 다시 수행합니다.
 - `timeout`:  사용자 입력을 대기할 시간(초)입니다.
 - `verify`:  자격 정보를 W&B 서버에서 검증합니다.
 - `referrer`:  URL 로그인 요청에 사용할 referrer입니다.




**반환(Returns):**
 
 - `bool`:  `key` 가 설정되어 있으면 True 를 반환합니다.



**예외(Raises):**
 
 - `AuthenticationError`:  `api_key` 가 서버에서 검증에 실패한 경우 발생합니다.
 - `UsageError`:  `api_key` 를 설정할 수 없고 tty(터미널)이 없는 경우 발생합니다.
```