import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx'

# box3d

<CTAButtons githubLink='https://github.com/wandb/wandb/blob/main/wandb/sdk/data_types/object_3d.py'/>




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

Returns a Box3D. 



**Args:**
 
 - `center`:  The center point of the box as a length-3 ndarray. 
 - `size`:  The box's X, Y and Z dimensions as a length-3 ndarray. 
 - `orientation`:  The rotation transforming global XYZ coordinates  into the box's local XYZ coordinates, given as a length-4  ndarray [r, x, y, z] corresponding to the non-zero quaternion  r + xi + yj + zk. 
 - `color`:  The box's color as an (r, g, b) tuple with `0 <= r,g,b <= 1`. 
 - `label`:  An optional label for the box. 
 - `score`:  An optional score for the box.