# save

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/wandb_run.py#L1879-L1985' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

1つ以上のファイルをW&Bに同期します。

```python
save(
    glob_str: Optional[Union[str, os.PathLike]] = None,
    base_path: Optional[Union[str, os.PathLike]] = None,
    policy: "PolicyName" = "live"
) -> Union[bool, List[str]]
```

相対パスは現在の作業ディレクトリーからの相対パスとなります。

Unixのglob（例えば "myfiles/*"）は、`save` が呼び出される時点で展開され、`policy` に関係なく適用されます。特に、新しいファイルは自動的にはピックアップされません。

アップロードされたファイルのディレクトリー構造を管理するために、`base_path` を指定することができます。それは `glob_str` のプレフィックスであり、その下のディレクトリー構造は保持されます。以下の例で説明します：

```
wandb.save("these/are/myfiles/*")
# => ファイルを run の "these/are/myfiles/" フォルダーに保存します。

wandb.save("these/are/myfiles/*", base_path="these")
# => ファイルを run の "are/myfiles/" フォルダーに保存します。

wandb.save("/User/username/Documents/run123/*.txt")
# => ファイルを run の "run123/" フォルダーに保存します。以下の注意を参照してください。

wandb.save("/User/username/Documents/run123/*.txt", base_path="/User")
# => ファイルを run の "username/Documents/run123/" フォルダーに保存します。

wandb.save("files/*/saveme.txt")
# => 各 "saveme.txt" ファイルを "files/" の適切なサブディレクトリーに保存します。
```

注: 絶対パスまたはglobが指定され、`base_path` がない場合、上記の例のように1つのディレクトリーレベルが保持されます。

| 引数 |  |
| :--- | :--- |
|  `glob_str` |  相対パスまたは絶対パス、またはUnixのglob。 |
|  `base_path` |  ディレクトリー構造を推測するためのパス。例を参照。 |
|  `policy` |  `live`、`now`、または `end` のいずれか。 * live: ファイルが変更されるたびにアップロードし、前のバージョンを上書きします * now: 現在のファイルを1回アップロードします * end: run が終了したときにファイルをアップロードします |

| 戻り値 |  |
| :--- | :--- |
|  マッチしたファイル用に作成されたシンボリックリンクのパス。過去の理由により、レガシーコードではブール値を返すことがあります。 |