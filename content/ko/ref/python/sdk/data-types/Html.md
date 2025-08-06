---
title: 'Html

  '
data_type_classification: class
menu:
  reference:
    identifier: ko-ref-python-sdk-data-types-Html
object_type: python_sdk_data_type
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/data_types/html.py >}}




## <kbd>class</kbd> `Html`
W&B에서 HTML 콘텐츠를 로그할 때 사용하는 클래스입니다.

### <kbd>method</kbd> `Html.__init__`

```python
__init__(
    data: Union[str, pathlib.Path, ForwardRef('TextIO')],
    inject: bool = True,
    data_is_not_path: bool = False
) → None
```

W&B HTML 오브젝트를 생성합니다.



**ARG:**
  data:  ".html" 확장자를 가진 파일 경로를 나타내는 문자열이거나, 또는 HTML 코드가 포함된 문자열 또는 IO 오브젝트입니다.
 - `inject`:  HTML 오브젝트에 스타일시트를 추가합니다. False로 설정하면 HTML이 변경 없이 그대로 전달됩니다.
 - `data_is_not_path`:  False로 설정하면 data가 파일 경로로 처리됩니다.



**예시:**
 파일 경로를 전달해서 초기화할 수 있습니다:

```python
with wandb.init() as run:
    run.log({"html": wandb.Html("./index.html")})
```

또는, 문자열이나 IO 오브젝트로 HTML 코드를 직접 전달해서 초기화할 수 있습니다:

```python
with wandb.init() as run:
    run.log({"html": wandb.Html("<h1>Hello, world!</h1>")})
```




---