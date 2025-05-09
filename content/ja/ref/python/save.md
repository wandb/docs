---
title: 保存
menu:
  reference:
    identifier: ja-ref-python-save
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L1875-L1979 >}}

1 つ以上のファイルを W&B に同期します。

```python
save(
    glob_str: (str | os.PathLike | None) = None,
    base_path: (str | os.PathLike | None) = None,
    policy: PolicyName = "live"
) -> (bool | list[str])
```

相対パスは現在の作業ディレクトリーに対するものです。

Unix のグロブ（例: "myfiles/*"）は、`save` が呼び出された時点で展開され、`policy` に関係ありません。特に、新しいファイルは自動的に取得されません。

アップロードされたファイルのディレクトリー構造を制御するために `base_path` を指定することができます。これは `glob_str` のプレフィックスであり、その下のディレクトリー構造は保持されます。以下の例で理解すると良いでしょう。

```
wandb.save("these/are/myfiles/*")
# => 保存されたファイルは run の "these/are/myfiles/" フォルダー内にあります。

wandb.save("these/are/myfiles/*", base_path="these")
# => 保存されたファイルは run の "are/myfiles/" フォルダー内にあります。

wandb.save("/User/username/Documents/run123/*.txt")
# => 保存されたファイルは run の "run123/" フォルダー内にあります。以下の注意点を参照してください。

wandb.save("/User/username/Documents/run123/*.txt", base_path="/User")
# => 保存されたファイルは run の "username/Documents/run123/" フォルダー内にあります。

wandb.save("files/*/saveme.txt")
# => 各 "saveme.txt" ファイルは "files/" の適切なサブディレクトリーに保存されます。
```

注意: 絶対パスやグロブが与えられ、`base_path` がない場合、例のように 1 つのディレクトリー レベルが保持されます。

| Args |  |
| :--- | :--- |
|  `glob_str` |  相対または絶対パス、または Unix グロブ。 |
|  `base_path` |  ディレクトリー構造を推測するためのパス; 例を参照してください。 |
|  `policy` |  `live`、`now`、または `end` のいずれか。 * live: ファイルが変更されるたびにアップロードし、以前のバージョンを上書きする * now: 現在一度だけファイルをアップロードする * end: run が終了したときにファイルをアップロードする |

| Returns |  |
| :--- | :--- |
|  一致したファイルに対して作成されたシンボリックリンクのパス。歴史的な理由により、レガシー コードではブール値を返すことがあります。 |