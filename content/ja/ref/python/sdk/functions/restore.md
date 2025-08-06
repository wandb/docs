---
title: 'restore()

  '
object_type: python_sdk_actions
data_type_classification: function
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_run.py >}}




### <kbd>function</kbd> `restore`

```python
restore(
    name: 'str',
    run_path: 'str | None' = None,
    replace: 'bool' = False,
    root: 'str | None' = None
) → None | TextIO
```

指定したファイルをクラウドストレージからダウンロードします。

ファイルは、現在のディレクトリーまたは run ディレクトリーに配置されます。デフォルトでは、そのファイルがまだ存在しない場合のみダウンロードします。



**引数:**
 
 - `name`:  ファイル名。 
 - `run_path`:  ファイルを取得する run のパス（例：`username/project_name/run_id`）。wandb.init が呼び出されていない場合は必須です。 
 - `replace`:  ファイルがすでにローカルに存在していても再ダウンロードするかどうか。 
 - `root`:  ファイルをダウンロードするディレクトリー。デフォルトは現在のディレクトリー、wandb.init が呼び出されていれば run ディレクトリー。



**戻り値:**
 ファイルが見つからなかった場合は None、見つかった場合は読み込み用にオープンされたファイルオブジェクト。



**例外:**
 
 - `CommError`:  W&B がバックエンドへ接続できない場合に発生。 
 - `ValueError`:  ファイルが見つからない場合や run_path を特定できない場合に発生。
```