
# login

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/wandb_login.py#L46-L103' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

W&Bのログイン資格情報を設定します。

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

デフォルトでは、W&Bサーバーでの認証を行わずに資格情報をローカルにのみ保存します。資格情報を確認するには、verify=True を渡します。

| 引数 |  |
| :--- | :--- |
|  `anonymous` |  (string, optional) "must", "allow", "never" のいずれかを指定できます。"must" が設定されている場合、常に匿名でログインし、"allow" が設定されている場合、ユーザーが既にログインしていない場合のみ匿名ユーザーを作成します。 |
|  `key` |  (string, optional) 認証キー。 |
|  `relogin` |  (bool, optional) trueに設定すると、再度APIキーの入力が促されます。 |
|  `host` |  (string, optional) 接続するホストのアドレス。 |
|  `force` |  (bool, optional) trueに設定すると、強制的に再ログインします。 |
|  `timeout` |  (int, optional) ユーザー入力を待つ秒数。 |
|  `verify` |  (bool) 資格情報をW&Bサーバーで確認します。 |

| 戻り値 |  |
| :--- | :--- |
|  `bool` |  キーが設定されている場合にtrueを返します。 |

| 例外 |  |
| :--- | :--- |
|  AuthenticationError - api_keyがサーバーでの認証に失敗した場合 UsageError - api_keyが設定できず、ttyが存在しない場合 |

