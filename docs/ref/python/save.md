
# save

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/wandb_run.py#L1879-L1985' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

1つ以上のファイルをW&Bに同期します。

```python
save(
    glob_str: Optional[Union[str, os.PathLike]] = None,
    base_path: Optional[Union[str, os.PathLike]] = None,
    policy: "PolicyName" = "live"
) -> Union[bool, List[str]]
```

相対パスは現在の作業ディレクトリーに対して相対的です。

"myfiles/*" のようなUnixグロブは、`policy` に関係なく `save` が呼び出された時点で展開されます。特に、新しいファイルは自動的に選択されません。

アップロードするファイルのディレクトリー構造を制御するために `base_path` を指定することができます。これは `glob_str` の接頭辞であるべきで、その下のディレクトリー構造が保持されます。以下の例で最もよく理解できます：

```
wandb.save("these/are/myfiles/*")
# => ファイルをrun内の "these/are/myfiles/" フォルダーに保存します。

wandb.save("these/are/myfiles/*", base_path="these")
# => ファイルをrun内の "are/myfiles/" フォルダーに保存します。

wandb.save("/User/username/Documents/run123/*.txt")
# => ファイルをrun内の "run123/" フォルダーに保存します。以下の注意を参照してください。

wandb.save("/User/username/Documents/run123/*.txt", base_path="/User")
# => ファイルをrun内の "username/Documents/run123/" フォルダーに保存します。

wandb.save("files/*/saveme.txt")
# => 各 "saveme.txt" ファイルを "files/" の適切なサブディレクトリーに保存します。
```

注意: 絶対パスまたはグロブが指定され、`base_path` がない場合、上記の例のように1つのディレクトリーレベルが保持されます。

| 引数 |  |
| :--- | :--- |
|  `glob_str` |  相対または絶対パスまたはUnixグロブ。 |
|  `base_path` |  ディレクトリー構造を推測するためのパス。例を参照。 |
|  `policy` |  `live`、`now`、`end` のいずれか。 * live: ファイルが変更されるたびにアップロード。以前のバージョンを上書きします。 * now: 今すぐにファイルを1回アップロード。 * end: runが終了したときにファイルをアップロード。 |

| 戻り値 |  |
| :--- | :--- |
|  マッチしたファイルのシンボリックリンクのパス。歴史的な理由により、レガシーコードではブーリアンを返すことがあります。 |