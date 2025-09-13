---
title: HTML
data_type_classification: class
menu:
  reference:
    identifier: ja-ref-python-sdk-data-types-Html
object_type: python_sdk_data_type
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/data_types/html.py >}}




## <kbd>クラス</kbd> `Html`
HTML コンテンツを W&B に ログ するための W&B のクラスです。 

### <kbd>メソッド</kbd> `Html.__init__`

```python
__init__(
    data: Union[str, pathlib.Path, ForwardRef('TextIO')],
    inject: bool = True,
    data_is_not_path: bool = False
) → None
```

W&B の HTML オブジェクトを作成します。 



**Args:**
  data:  拡張子 ".html" のファイルへのパスを表す文字列、またはリテラルな HTML を含む文字列もしくは IO オブジェクト。 
 - `inject`:  HTML オブジェクトにスタイルシートを追加します。False に設定すると、HTML は変更されません。 
 - `data_is_not_path`:  False の場合、data はファイルへのパスとして扱われます。 



**Examples:**
 ファイルへのパスを渡して初期化できます: 

```python
with wandb.init() as run:
    run.log({"html": wandb.Html("./index.html")})
``` 

また、文字列または IO オブジェクトでリテラルな HTML を渡して初期化することもできます: 

```python
with wandb.init() as run:
    run.log({"html": wandb.Html("<h1>Hello, world!</h1>")})
``` 




---