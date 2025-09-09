---
title: Object3D
data_type_classification: class
menu:
  reference:
    identifier: ja-ref-python-sdk-data-types-Object3D
object_type: python_sdk_data_type
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/data_types/object_3d.py >}}




## <kbd>クラス</kbd> `Object3D`
W&B の 3D 点群用クラスです。

### <kbd>メソッド</kbd> `Object3D.__init__`

```python
__init__(
    data_or_path: Union[ForwardRef('np.ndarray'), str, pathlib.Path, ForwardRef('TextIO'), dict],
    caption: Optional[str] = None,
    **kwargs: Optional[str, ForwardRef('FileFormat3D')]
) → None
```

W&B Object3D オブジェクトを作成します。



**Args:**
 
 - `data_or_path`:  Object3D は、ファイルまたは NumPy 配列から初期化できます。 
 - `caption`:  表示のためにそのオブジェクトに関連付けられるキャプション。 



**例:**
 NumPy 配列の形状は次のいずれかである必要があります

```text
[[x y z],       ...] nx3
[[x y z c],     ...] nx4 where c はカテゴリで、サポートされる範囲は [1, 14]
[[x y z r g b], ...] nx6 where rgb は色
``` 




---