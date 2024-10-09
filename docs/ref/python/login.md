# login

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_login.py#L46-L104' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

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

기본적으로, 이것은 자격 증명을 W&B 서버에 확인하지 않고 로컬에만 저장합니다. 자격 증명을 확인하려면 `verify=True`를 전달하세요.

| 인수 |  |
| :--- | :--- |
|  `anonymous` |  (문자열, 선택 사항) "must", "allow", "never" 중 하나일 수 있습니다. "must"로 설정하면 항상 사용자를 익명으로 로그인합니다. "allow"로 설정하면 사용자가 이미 로그인되어 있지 않은 경우에만 익명 사용자를 생성합니다. "never"로 설정하면 결코 사용자를 익명으로 로그인하지 않습니다. 기본값은 "never"로 설정되어 있습니다. |
|  `key` |  (문자열, 선택 사항) 사용할 API 키. |
|  `relogin` |  (bool, 선택 사항) true인 경우, API 키를 다시 입력하라는 메시지를 표시합니다. |
|  `host` |  (문자열, 선택 사항) 연결할 서버. |
|  `force` |  (bool, 선택 사항) true인 경우, 다시 로그인을 강제합니다. |
|  `timeout` |  (int, 선택 사항) 사용자 입력을 대기하는 초 수. |
|  `verify` |  (bool) W&B 서버에서 자격 증명을 확인합니다. |

| 반환값 |  |
| :--- | :--- |
|  `bool` |  키가 설정된 경우 |

| 발생 예외 |  |
| :--- | :--- |
|  AuthenticationError - api_key가 서버에서 확인에 실패할 경우 UsageError - api_key를 설정할 수 없고 tty가 없는 경우 |