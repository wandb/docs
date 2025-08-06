---
data_type_classification: class
menu:
  reference:
    identifier: ja-ref-python-sdk-data-types-Html
object_type: python_sdk_data_type
title: Html
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/data_types/html.py >}}




## <kbd>class</kbd> `Html`
W&B class for logging HTML content to W&B. 

### <kbd>method</kbd> `Html.__init__`

```python
__init__(
    data: Union[str, pathlib.Path, ForwardRef('TextIO')],
    inject: bool = True,
    data_is_not_path: bool = False
) → None
```

Creates a W&B HTML object. 



**Args:**
  data:  A string that is a path to a file with the extension ".html",  or a string or IO object containing literal HTML. 
 - `inject`:  Add a stylesheet to the HTML object. If set  to False the HTML will pass through unchanged. 
 - `data_is_not_path`:  If set to False, the data will be  treated as a path to a file. 



**Examples:**
 It can be initialized by providing a path to a file: 

```python
with wandb.init() as run:
    run.log({"html": wandb.Html("./index.html")})
``` 

Alternatively, it can be initialized by providing literal HTML, in either a string or IO object: 

```python
with wandb.init() as run:
    run.log({"html": wandb.Html("<h1>Hello, world!</h1>")})
``` 




---