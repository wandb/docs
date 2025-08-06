---
title: Object3D
data_type_classification: class
menu:
  reference:
    identifier: ko-ref-python-sdk-data-types-Object3D
object_type: python_sdk_data_type
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/data_types/object_3d.py >}}




## <kbd>class</kbd> `Object3D`
W&B에서 3D 포인트 클라우드를 위한 클래스입니다. 

### <kbd>method</kbd> `Object3D.__init__`

```python
__init__(
    data_or_path: Union[ForwardRef('np.ndarray'), str, pathlib.Path, ForwardRef('TextIO'), dict],
    caption: Optional[str] = None,
    **kwargs: Optional[str, ForwardRef('FileFormat3D')]
) → None
```

W&B Object3D 오브젝트를 생성합니다.



**ARG:**
 
 - `data_or_path`:  Object3D는 파일이나 numpy 배열로 초기화할 수 있습니다.
 - `caption`:  오브젝트에 표시할 캡션입니다.



**예시:**
 numpy 배열의 형태는 아래 중 하나여야 합니다.

```text
[[x y z],       ...] nx3
[[x y z c],     ...] nx4 (여기서 c는 [1, 14] 범위의 카테고리)
[[x y z r g b], ...] nx6 (여기서 rgb는 컬러)
``` 




---