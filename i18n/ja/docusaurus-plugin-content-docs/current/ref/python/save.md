
# 保存

[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)GitHubでソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/wandb_run.py#L1752-L1782)
`glob_str`と一致するすべてのファイルが指定されたポリシーでwandbに同期されることを確認してください。

```python
save(
 glob_str: Optional[str] = None,
 base_path: Optional[str] = None,
 policy: "PolicyName" = "live"
) -> Union[bool, List[str]]
```
| 引数 |  |
| :--- | :--- |
| `glob_str` | (文字列) Unixグロブまたは通常のパスへの相対パスまたは絶対パス。これが指定されていない場合、メソッドは何も操作しません。|
| `base_path` | (文字列) グロブを実行するベースパス |
| `policy` | (文字列) `live`、`now`、`end` のいずれか - live: ファイルが変更されるたびにアップロードして、前のバージョンを上書きする - now: ファイルを今すぐ一度だけアップロードする - end: runが終了したときにのみファイルをアップロードする |