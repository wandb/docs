---
title: Object3D
object_type: python_sdk_data_type
data_type_classification: class
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/data_types/object_3d.py >}}




## <kbd>class</kbd> `Object3D`
W&B class for 3D point clouds. 

### <kbd>method</kbd> `Object3D.__init__`

```python
__init__(
    data_or_path: Union[ForwardRef('np.ndarray'), str, pathlib.Path, ForwardRef('TextIO'), dict],
    caption: Optional[str] = None,
    **kwargs: Optional[str, ForwardRef('FileFormat3D')]
) → None
```

Creates a W&B Object3D object. 



**Args:**
 
 - `data_or_path`:  Object3D can be initialized from a file or a numpy array. 
 - `caption`:  Caption associated with the object for display. 



**Examples:**
 The shape of the numpy array must be one of either 

```text
[[x y z],       ...] nx3
[[x y z c],     ...] nx4 where c is a category with supported range [1, 14]
[[x y z r g b], ...] nx6 where is rgb is color
``` 




---





