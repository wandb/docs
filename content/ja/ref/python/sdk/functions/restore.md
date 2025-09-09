---
title: restore()
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

指定したファイルをクラウド ストレージからダウンロードします。 

ファイルはカレント ディレクトリー、または run ディレクトリーに配置されます。デフォルトでは、ローカルに存在しない場合のみダウンロードします。 



**引数:**
 
 - `name`:  ファイル名。 
 - `run_path`:  ファイルを取得する対象の run への任意のパス。例: `username/project_name/run_id`。wandb.init が呼び出されていない場合は必須。 
 - `replace`:  ローカルに既に存在する場合でもファイルをダウンロードするかどうか。 
 - `root`:  ファイルのダウンロード先ディレクトリー。デフォルトはカレント ディレクトリー、または wandb.init が呼び出されている場合は run ディレクトリー。 



**戻り値:**
 ファイルが見つからない場合は None、見つかった場合は読み取り用に開かれたファイル オブジェクト。 



**例外:**
 
 - `CommError`:  W&B が W&B バックエンドに接続できない場合。 
 - `ValueError`:  ファイルが見つからない、または run_path を見つけられない場合。