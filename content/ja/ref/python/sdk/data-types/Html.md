---
title: Html
data_type_classification: class
menu:
  reference:
    identifier: ja-ref-python-sdk-data-types-Html
object_type: python_sdk_data_type
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/data_types/html.py >}}




## <kbd>class</kbd> `Html`
W&B で HTML コンテンツをログするためのクラスです。

### <kbd>method</kbd> `Html.__init__`

```python
__init__(
    data: Union[str, pathlib.Path, ForwardRef('TextIO')],
    inject: bool = True,
    data_is_not_path: bool = False
) → None
```

W&B の HTML オブジェクトを作成します。



**引数:**
  data:  ".html" 拡張子を持つファイルへのパス文字列、または HTML のリテラルを含む文字列や IO オブジェクト。
 - `inject`:  スタイルシートを HTML オブジェクトに追加します。False に設定すると HTML はそのまま渡されます。
 - `data_is_not_path`:  False の場合、data はファイルへのパスとして扱われます。



**使用例:**
 ファイルへのパスを指定して初期化できます:

```python
with wandb.init() as run:
    run.log({"html": wandb.Html("./index.html")})
```

または、リテラル HTML（文字列や IO オブジェクト）を直接指定して初期化することも可能です:

```python
with wandb.init() as run:
    run.log({"html": wandb.Html("<h1>Hello, world!</h1>")})
```




---