# login

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/wandb_login.py#L46-L103' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

W&Bのログイン認証情報を設定します。

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

デフォルトでは、認証情報はW&Bサーバーで検証されることなくローカルにのみ保存されます。認証情報を検証するには、verify=Trueを渡してください。

| 引数 |  |
| :--- | :--- |
|  `anonymous` |  (string, optional) "must"、"allow"、"never"のいずれかを指定できます。"must"が設定されると、常に匿名でログインします。"allow"が設定されると、ユーザーが既にログインしていない場合にのみ匿名ユーザーを作成します。 |
|  `key` |  (string, optional) 認証キー。 |
|  `relogin` |  (bool, optional) trueに設定すると、APIキーの再入力を求めます。 |
|  `host` |  (string, optional) 接続するホスト。 |
|  `force` |  (bool, optional) trueに設定すると、強制的に再ログインします。 |
|  `timeout` |  (int, optional) ユーザー入力を待つ秒数。 |
|  `verify` |  (bool) W&Bサーバーで認証情報を検証します。 |

| 戻り値 |  |
| :--- | :--- |
|  `bool` |  キーが設定されている場合 |

| 例外 |  |
| :--- | :--- |
|  AuthenticationError - api_keyのサーバーでの検証に失敗した場合 UsageError - api_keyが設定できず、ttyがない場合 |
