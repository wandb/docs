---
title: 'restore()

  '
data_type_classification: function
menu:
  reference:
    identifier: ja-ref-python-sdk-functions-restore
object_type: python_sdk_actions
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

ファイルはカレントディレクトリーまたは run のディレクトリーに配置されます。デフォルトでは、ファイルがまだ存在しない場合のみダウンロードします。



**引数:**
 
 - `name`:  ファイル名。
 - `run_path`:  ファイルを取得する run のパス（例: `username/project_name/run_id`）。wandb.init が呼ばれていない場合は必須です。
 - `replace`:  ローカルにすでにファイルが存在している場合でも再度ダウンロードするかどうか。
 - `root`:  ファイルをダウンロードするディレクトリー。デフォルトは現在のディレクトリー、または wandb.init が呼ばれていれば run ディレクトリーです。



**戻り値:**
 ファイルが見つからない場合は None、それ以外の場合は読み取り用のファイルオブジェクト。



**例外:**
 
 - `CommError`:  W&B が W&B バックエンドに接続できない場合。
 - `ValueError`:  ファイルが見つからない、または run_path が見つからない場合。