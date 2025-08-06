---
title: Molecule
data_type_classification: class
menu:
  reference:
    identifier: ja-ref-python-sdk-data-types-Molecule
object_type: python_sdk_data_type
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
 
 - `data_or_path`:  ファイル名または io オブジェクトから Molecule を初期化できます。
 - `caption`:  分子の表示用キャプション。



---