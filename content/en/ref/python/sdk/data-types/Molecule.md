---
title: Molecule
namespace: python_sdk_data_type
python_object_type: class
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/data_types/molecule.py >}}




## <kbd>class</kbd> `Molecule`
W&B class for 3D Molecular data. 

### <kbd>method</kbd> `Molecule.__init__`

```python
__init__(
    data_or_path: Union[str, pathlib.Path, ForwardRef('TextIO')],
    caption: Optional[str] = None,
    **kwargs: str
) → None
```

Initialize a Molecule object. 



**Args:**
 
 - `data_or_path`:  Molecule can be initialized from a file name or an io object. 
 - `caption`:  Caption associated with the molecule for display. 




---




