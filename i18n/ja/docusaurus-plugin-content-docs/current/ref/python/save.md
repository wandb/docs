# 保存



[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)View source on GitHub](https://www.github.com/wandb/client/tree/c505c66a5f9c1530671564dae3e9e230f72f6584/wandb/sdk/wandb_run.py#L1714-L1743)



指定されたポリシーで、'glob_str'と一致するすべてのファイルをwandbと同期化します。

```python
save(
 glob_str: Optional[str] = None,
 base_path: Optional[str] = None,
 policy: "PolicyName" = "live"
) -> Union[bool, List[str]]
```





| 引数 | |
| :--- | :--- |
| `glob_str` | （文字列）unix globや正規パスへの相対パスまたは絶対パス。指定しない場合、メソッドはnoopになります。 |
| `base_path` | （文字列）globを実行する際のベースパス |
| `policy` | （文字列）`live`、`now`、または`end` - live：変更されたファイルをアップロードし、前のバージョンを上書きします - now：ファイルをすぐに1回アップロードします - end：runの終了時にのみファイルをアップロードします |

