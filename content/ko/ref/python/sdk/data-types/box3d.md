---
title: 'box3d()

  '
data_type_classification: function
menu:
  reference:
    identifier: ko-ref-python-sdk-data-types-box3d
object_type: python_sdk_data_type
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/data_types/object_3d.py >}}




### <kbd>function</kbd> `box3d`

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

Box3D 를 반환합니다. 



**인자:**
 
 - `center`:  박스의 중심점을 길이 3인 ndarray 로 전달합니다. 
 - `size`:  박스의 X, Y, Z 크기를 길이 3인 ndarray 로 전달합니다. 
 - `orientation`:  전체 XYZ 좌표계를 박스의 로컬 XYZ 좌표계로 변환하는 회전을, 길이 4인 ndarray [r, x, y, z] 형태로 전달합니다. 이는 0 이 아닌 쿼터니언 r + xi + yj + zk 를 의미합니다. 
 - `color`:  박스의 색상을 (r, g, b) 튜플로 전달하며, 각 값은 0 <= r,g,b <= 1 입니다. 
 - `label`:  박스에 대한 선택적 라벨입니다. 
 - `score`:  박스에 대한 선택적 점수입니다.