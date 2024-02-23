
# 로그인

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/wandb_login.py#L46-L97' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>GitHub에서 소스 보기</a></button></p>


W&B 로그인 자격 증명 설정하기.

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

기본적으로, 이것은 W&B 서버와 확인하지 않고 로컬에만 자격 증명을 저장합니다. 자격 증명을 확인하려면, verify=True를 전달하세요.

| 인수 |  |
| :--- | :--- |
|  `anonymous` |  (문자열, 선택적) "must", "allow", 또는 "never"가 될 수 있습니다. "must"로 설정하면 항상 익명으로 로그인하고, "allow"로 설정하면 사용자가 이미 로그인하지 않은 경우에만 익명 사용자를 생성합니다. |
|  `key` |  (문자열, 선택적) 인증 키. |
|  `relogin` |  (불리언, 선택적) 참이면, API 키를 다시 요청합니다. |
|  `host` |  (문자열, 선택적) 연결할 호스트. |
|  `force` |  (불리언, 선택적) 참이면, 로그인을 강제로 다시 실행합니다. |
|  `timeout` |  (정수, 선택적) 사용자 입력을 기다리는 초 수. |
|  `verify` |  (불리언) W&B 서버와 자격 증명을 확인합니다. |

| 반환값 |  |
| :--- | :--- |
|  `bool` |  키가 구성되면 |

| 예외 |  |
| :--- | :--- |
|  인증 오류 - api_key가 서버와의 검증에 실패할 경우 사용 오류 - api_key를 구성할 수 없고 tty가 없는 경우