---
title: オブジェクト 3D
object_type: python_sdk_data_type
data_type_classification: class
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/data_types/object_3d.py >}}




## <kbd>class</kbd> `Object3D`
W&B の3Dポイントクラウド用クラスです。

### <kbd>method</kbd> `Object3D.__init__`

```python
__init__(
    data_or_path: Union[ForwardRef('np.ndarray'), str, pathlib.Path, ForwardRef('TextIO'), dict],
    caption: Optional[str] = None,
    **kwargs: Optional[str, ForwardRef('FileFormat3D')]
) → None
```

W&B の Object3D オブジェクトを作成します。

**引数:**
 
 - `data_or_path`:  Object3D はファイルまたは numpy 配列から初期化できます。
 - `caption`:  表示用のキャプションをオブジェクトに関連付けます。

**例:**
 numpy 配列の形状は、以下のいずれかである必要があります。

```text
[[x y z],       ...] nx3
[[x y z c],     ...] nx4   ※cは1～14のカテゴリ
[[x y z r g b], ...] nx6   ※rgbは色情報
``` 




---