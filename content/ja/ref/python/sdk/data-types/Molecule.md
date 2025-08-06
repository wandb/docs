---
title: 分子
object_type: python_sdk_data_type
data_type_classification: class
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/data_types/molecule.py >}}




## <kbd>class</kbd> `Molecule`
W&B の 3D 分子データ用クラスです。

### <kbd>method</kbd> `Molecule.__init__`

```python
__init__(
    data_or_path: Union[str, pathlib.Path, ForwardRef('TextIO')],
    caption: Optional[str] = None,
    **kwargs: str
) → None
```

Molecule オブジェクトを初期化します。



**引数:**
 
 - `data_or_path`: Molecule はファイル名または io オブジェクトから初期化できます。
 - `caption`: 表示用の Molecule に関連するキャプション。




---