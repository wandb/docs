---
title: box3d()
data_type_classification: function
menu:
  reference:
    identifier: ja-ref-python-sdk-data-types-box3d
object_type: python_sdk_data_type
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/data_types/object_3d.py >}}




### <kbd>関数</kbd> `box3d`

```python
box3d(
    center: 'npt.ArrayLike',
    size: 'npt.ArrayLike',
    orientation: 'npt.ArrayLike',
    color: 'RGBColor',
    label: 'Optional[str]' = None,
    score: 'Optional[numeric]' = None
) → Box3D
```

Box3D を返します。 



**引数:**
 
 - `center`:  ボックスの中心点 (長さ 3 の ndarray)。 
 - `size`:  ボックスの X、Y、Z 各次元の大きさ (長さ 3 の ndarray)。 
 - `orientation`:  グローバル XYZ 座標を ボックスのローカル XYZ 座標に変換する回転。0 ではない四元数 r + xi + yj + zk に対応する、長さ 4 の ndarray [r, x, y, z] で与えます。 
 - `color`:  ボックスの色を 0 <= r,g,b <= 1 の (r, g, b) タプルで指定します。 
 - `label`:  ボックスの任意のラベル。 
 - `score`:  ボックスの任意のスコア。