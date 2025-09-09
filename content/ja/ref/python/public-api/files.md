---
title: ファイル
data_type_classification: module
menu:
  reference:
    identifier: ja-ref-python-public-api-files
object_type: public_apis_namespace
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/apis/public/files.py >}}




# <kbd>モジュール</kbd> `wandb.apis.public`
W&B の File オブジェクト向け Public API。 

このモジュールは、W&B に保存されたファイルを操作するためのクラスを提供します。 



**例:**
 ```python
from wandb.apis.public import Api

# 特定の run からファイルを取得
run = Api().run("entity/project/run_id")
files = run.files()

# ファイルを操作する
for file in files:
     print(f"File: {file.name}")
     print(f"Size: {file.size} bytes")
     print(f"Type: {file.mimetype}")

     # ファイルをダウンロード
     if file.size < 1000000:  # 1MB 未満
         file.download(root="./downloads")

     # 大きなファイル用に S3 URI を取得
     if file.size >= 1000000:
         print(f"S3 URI: {file.path_uri}")
``` 



**注:**

> このモジュールは W&B Public API の一部で、W&B に保存されたファイルへの アクセス、ダウンロード、管理 のためのメソッドを提供します。ファイルは通常、特定の run に紐づいており、モデルの重み、データセット、可視化、その他のアーティファクトを含むことがあります。 

## <kbd>class</kbd> `Files`
`File` オブジェクトのコレクションに対する遅延イテレーター。 

run 中に W&B にアップロードされたファイルへ アクセスして管理 します。大量のファイルを反復処理する際は、ページネーションを自動で処理します。 



**例:**
 ```python
from wandb.apis.public.files import Files
from wandb.apis.public.api import Api

# 例となる run オブジェクト
run = Api().run("entity/project/run-id")

# run 内のファイルを反復処理するための Files オブジェクトを作成
files = Files(api.client, run)

# ファイルを反復処理
for file in files:
     print(file.name)
     print(file.url)
     print(file.size)

     # ファイルをダウンロード
     file.download(root="download_directory", replace=True)
``` 

### <kbd>メソッド</kbd> `Files.__init__`

```python
__init__(
    client: 'RetryingClient',
    run: 'Run',
    names: 'list[str] | None' = None,
    per_page: 'int' = 50,
    upload: 'bool' = False,
    pattern: 'str | None' = None
)
```

`File` オブジェクトのコレクションに対する遅延イテレーターを初期化します。 

ファイルは必要に応じて W&B サーバーからページ単位で取得されます。 



**引数:**
 client: ファイルを含む run オブジェクト run: ファイルを含む run オブジェクト names (list, オプション): ファイルをフィルタするためのファイル名のリスト per_page (int, オプション): 1 ページあたりに取得するファイル数 upload (bool, オプション): `True` の場合、各ファイルのアップロード URL を取得します pattern (str, オプション): W&B からファイルを返す際にマッチさせるパターン。 このパターンは MySQL の LIKE 構文を使用します。 たとえば .json で終わるすべてのファイルにマッチさせるには "%.json" です。 names と pattern の両方が指定された場合は ValueError が送出されます。 


---


### <kbd>プロパティ</kbd> Files.length





---