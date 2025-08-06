---
title: ファイル
object_type: public_apis_namespace
data_type_classification: module
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/apis/public/files.py >}}




# <kbd>module</kbd> `wandb.apis.public`
W&B Public API で File オブジェクトを扱うためのモジュールです。

このモジュールは、W&B 上に保存されたファイルを操作するためのクラスを提供します。



**例:**
 ```python
from wandb.apis.public import Api

# 特定の run からファイルを取得
run = Api().run("entity/project/run_id")
files = run.files()

# ファイルを操作
for file in files:
     print(f"File: {file.name}")
     print(f"Size: {file.size} bytes")
     print(f"Type: {file.mimetype}")

     # ファイルをダウンロード
     if file.size < 1000000:  # 1MB 未満の場合
         file.download(root="./downloads")

     # 大きいファイルの場合は S3 URI を取得
     if file.size >= 1000000:
         print(f"S3 URI: {file.path_uri}")
```



**注意:**

> このモジュールは W&B Public API の一部で、W&B に保存されているファイルへの アクセス、ダウンロード、および管理を行うメソッドを提供します。ファイルは通常、特定の Runs に紐付けられていて、モデルの重み、Datasets、可視化、その他の Artifacts などが含まれます。

## <kbd>class</kbd> `Files`
`File` オブジェクトのイテラブルなコレクションです。

Run 中に W&B へアップロードされたファイルの アクセス・管理ができます。大きなファイルコレクションを順に処理する際も、自動的にページネーションを行います。



**例:**
 ```python
from wandb.apis.public.files import Files
from wandb.apis.public.api import Api

# run オブジェクトの例
run = Api().run("entity/project/run-id")

# run 内のファイルをイテレートするための Files オブジェクトを作成
files = Files(api.client, run)

# ファイルをイテレート
for file in files:
     print(file.name)
     print(file.url)
     print(file.size)

     # ファイルをダウンロード
     file.download(root="download_directory", replace=True)
``` 

### <kbd>method</kbd> `Files.__init__`

```python
__init__(client, run, names=None, per_page=50, upload=False)
```

特定の run に対する `File` オブジェクトのイテラブルなコレクションを作成します。



**引数:**
 client: ファイルを含む run オブジェクト run: ファイルを含む run オブジェクト names (list, オプション): ファイル名でファイルを絞り込むリスト per_page (int, オプション): 1ページあたりに取得するファイル数 upload (bool, オプション): `True` の場合、各ファイルのアップロードURLも取得する 


---


### <kbd>property</kbd> Files.length





---