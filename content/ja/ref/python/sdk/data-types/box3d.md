---
title: box3d()
object_type: python_sdk_data_type
data_type_classification: function
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

 - `center`: ボックスの中心点を表す長さ3の ndarray です。
 - `size`: ボックスの X, Y, Z の各次元を表す長さ3の ndarray です。
 - `orientation`: グローバル XYZ 座標をボックスのローカル XYZ 座標へ変換する回転を、長さ4の ndarray [r, x, y, z] で指定します（これは非ゼロクォータニオン r + xi + yj + zk に対応します）。
 - `color`: ボックスの色を (r, g, b) タプルで指定します。各値は 0 <= r,g,b <= 1 の範囲です。
 - `label`: ボックスに付けるオプションラベルです。
 - `score`: ボックスに付けるオプションのスコアです。