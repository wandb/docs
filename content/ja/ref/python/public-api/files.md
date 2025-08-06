---
title: ファイル
data_type_classification: module
menu:
  reference:
    identifier: ja-ref-python-public-api-files
object_type: public_apis_namespace
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/apis/public/files.py >}}




# <kbd>module</kbd> `wandb.apis.public`
W&B Public API による File オブジェクトの操作

このモジュールは、W&B に保存されたファイルとやり取りするためのクラスを提供します。



**例:**
 ```python
from wandb.apis.public import Api

# 特定の Run からファイルを取得
run = Api().run("entity/project/run_id")
files = run.files()

# ファイルを扱う
for file in files:
     print(f"File: {file.name}")
     print(f"Size: {file.size} bytes")
     print(f"Type: {file.mimetype}")

     # ファイルのダウンロード
     if file.size < 1000000:  # 1MB 未満
         file.download(root="./downloads")

     # 大きなファイルの場合は S3 URI を取得
     if file.size >= 1000000:
         print(f"S3 URI: {file.path_uri}")
``` 



**注意:**

> このモジュールは W&B Public API の一部であり、W&B に保存されたファイルへのアクセス・ダウンロード・管理を行うメソッドを提供します。ファイルは通常、特定の Run に紐づいており、モデルの重み・データセット・可視化・その他の Artifacts などが含まれます。

## <kbd>class</kbd> `Files`
`File` オブジェクトのイテラブルコレクション

Run 中に W&B にアップロードされたファイルのアクセス・管理を行います。大量のファイルでも自動的にページネーション処理を行います。



**例:**
 ```python
from wandb.apis.public.files import Files
from wandb.apis.public.api import Api

# 例：Run オブジェクトの作成
run = Api().run("entity/project/run-id")

# Run 内のファイルをイテレートする Files オブジェクトの作成
files = Files(api.client, run)

# ファイルのイテレーション
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

特定の Run に紐づいた `File` オブジェクトのイテラブルコレクション。



**引数:**
 client: ファイルを含む run オブジェクト run: ファイルを含む run オブジェクト names (list, オプション): 対象のファイル名リストでフィルタ per_page (int, オプション): 1 ページあたりに取得するファイル数 upload (bool, オプション): `True` の場合は各ファイルのアップロード用 URL も取得します


---


### <kbd>property</kbd> Files.length





---