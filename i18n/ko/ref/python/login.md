
# 로그인

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/sdk/wandb_login.py#L46-L97' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>GitHub에서 소스 보기</a></button></p>


W&B 로그인 자격증명 설정하기.

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

기본적으로, 이것은 W&B 서버로 자격증명을 확인하지 않고 로컬에만 자격증명을 저장합니다. 자격증명을 확인하고 싶다면, verify=True를 전달하세요.

| 인수 |  |
| :--- | :--- |
|  `anonymous` |  (문자열, 선택사항) "must", "allow", 또는 "never"가 될 수 있습니다. "must"로 설정하면 항상 익명으로 로그인하며, "allow"로 설정하면 사용자가 이미 로그인되어 있지 않은 경우에만 익명 사용자를 생성합니다. |
|  `key` |  (문자열, 선택사항) 인증 키입니다. |
|  `relogin` |  (불린, 선택사항) 참이면, API 키를 다시 입력하라는 메시지가 나타납니다. |
|  `host` |  (문자열, 선택사항) 연결할 호스트입니다. |
|  `force` |  (불린, 선택사항) 참이면, 재로그인을 강제합니다. |
|  `timeout` |  (정수, 선택사항) 사용자 입력을 기다리는 시간(초)입니다. |
|  `verify` |  (불린) W&B 서버와 자격증명을 확인합니다. |

| 반환값 |  |
| :--- | :--- |
|  `bool` |  키가 구성된 경우 |

| 예외 |  |
| :--- | :--- |
|  AuthenticationError - 서버와의 인증 키 확인 실패 시 발생 UsageError - API 키를 구성할 수 없고 tty가 없는 경우 발생 |