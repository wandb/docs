---
title: save
menu:
  reference:
    identifier: ja-ref-python-save
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L1875-L1979 >}}

1つまたは複数のファイルを W&B に同期します。

```python
save(
    glob_str: (str | os.PathLike | None) = None,
    base_path: (str | os.PathLike | None) = None,
    policy: PolicyName = "live"
) -> (bool | list[str])
```

相対パスは、現在の作業ディレクトリーからの相対パスです。

"myfiles/*" などの Unix glob は、`policy` に関係なく、`save` が呼び出された時点で展開されます。特に、新しいファイルは自動的に取得されません。

`base_path` を指定して、アップロードされたファイルのディレクトリー構造を制御できます。これは `glob_str` のプレフィックスである必要があり、その下のディレクトリー構造が保持されます。これは、以下の例で理解するのが最適です。

```
wandb.save("these/are/myfiles/*")
# => run の "these/are/myfiles/" フォルダーにファイルを保存します。

wandb.save("these/are/myfiles/*", base_path="these")
# => run の "are/myfiles/" フォルダーにファイルを保存します。

wandb.save("/User/username/Documents/run123/*.txt")
# => run の "run123/" フォルダーにファイルを保存します。下記の注記を参照してください。

wandb.save("/User/username/Documents/run123/*.txt", base_path="/User")
# => run の "username/Documents/run123/" フォルダーにファイルを保存します。

wandb.save("files/*/saveme.txt")
# => 各 "saveme.txt" ファイルを "files/" の適切なサブディレクトリーに保存します。
```

注: 絶対パスまたは glob が指定され、`base_path` が指定されていない場合、上記の例のように 1 つのディレクトリー・レベルが保持されます。

| Args |  |
| :--- | :--- |
| `glob_str` | 相対パスまたは絶対パス、あるいは Unix glob。 |
| `base_path` | ディレクトリー構造を推測するために使用するパス。例を参照してください。 |
| `policy` | `live`、`now`、または `end` のいずれか。 * live: ファイルが変更されるとアップロードし、以前のバージョンを上書きします * now: ファイルを今すぐ 1 回アップロードします * end: run の終了時にファイルをアップロードします |

| 戻り値 |  |
| :--- | :--- |
| マッチしたファイルに対して作成されたシンボリック・リンクへのパス。歴史的な理由から、レガシー・コードではブール値を返すことがあります。 |
