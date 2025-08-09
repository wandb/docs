---
title: 몰리큘
data_type_classification: class
menu:
  reference:
    identifier: ko-ref-python-sdk-data-types-Molecule
object_type: python_sdk_data_type
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/data_types/molecule.py >}}




## <kbd>class</kbd> `Molecule`
W&B의 3D 분자 데이터 전용 클래스입니다.

### <kbd>method</kbd> `Molecule.__init__`

```python
__init__(
    data_or_path: Union[str, pathlib.Path, ForwardRef('TextIO')],
    caption: Optional[str] = None,
    **kwargs: str
) → None
```

Molecule 오브젝트를 초기화합니다.



**ARG:**

 - `data_or_path`:  Molecule 은 파일 이름 또는 io 오브젝트로 초기화할 수 있습니다.
 - `caption`:  분자와 함께 표시할 캡션입니다.




---