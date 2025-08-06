---
title: 'Html

  '
object_type: python_sdk_data_type
data_type_classification: class
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
  data:  拡張子が ".html" のファイルのパスとなる文字列、または HTML のリテラルが含まれた文字列や IO オブジェクトです。
 - `inject`:  スタイルシートを HTML オブジェクトに追加します。False に設定すると、HTML は変更されずにそのまま渡されます。
 - `data_is_not_path`:  False の場合、data はファイルのパスとして扱われます。



**使用例:**
 ファイルパスを渡して初期化することができます:

```python
with wandb.init() as run:
    run.log({"html": wandb.Html("./index.html")})
```

または、リテラル HTML を文字列や IO オブジェクトで渡して初期化することもできます:

```python
with wandb.init() as run:
    run.log({"html": wandb.Html("<h1>Hello, world!</h1>")})
```




---