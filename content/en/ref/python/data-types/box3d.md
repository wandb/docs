---
title: box3d()
namespace: python_sdk_data_type
python_object_type: function
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

Returns a Box3D. 



**Args:**
 
 - `size`:  The box's X, Y and Z dimensions as a length-3 ndarray. 
 - `orientation`:  The rotation transforming global XYZ coordinates  into the box's local XYZ coordinates, given as a length-4  ndarray [r, x, y, z] corresponding to the non-zero quaternion  r + xi + yj + zk. 
 - `color`:  The box's color as an (r, g, b) tuple with 0 <= r,g,b <= 1. 
 - `label`:  An optional label for the box. 
 - `score`:  An optional score for the box. 



**Example:**
 The following example creates a point cloud with 60 boxes rotating around the X, Y and Z axes.  

```python
import wandb    

import math
import numpy as np
from scipy.spatial.transform import Rotation


with wandb.init() as run:
    run.log({
         "points": wandb.Object3D.from_point_cloud(
             points=np.random.uniform(-5, 5, size=(100, 3)),
             boxes=[
                 wandb.box3d(
                     center=(0.3*t - 3, 0, 0),
                     size=(0.1, 0.1, 0.1),
                     orientation=Rotation.from_euler('xyz', [t*math.pi/10, 0, 0]).as_quat(),
                     color=(0.5 + t/40, 0.5, 0.5),
                     label=f"box {t}",
                 )
                 for t in range(20)
             ]+[
                 wandb.box3d(
                     center=(0, 0.3*t - 3, 0.3),
                     size=(0.1, 0.1, 0.1),
                     orientation=Rotation.from_euler('xyz', [0, t*math.pi/10, 0]).as_quat(),
                     color=(0.5, 0.5 + t/40, 0.5),
                     label=f"box {t}",
                 )
                 for t in range(20)
             ]+[
                 wandb.box3d(
                     center=(0.3, 0.3, 0.3*t - 3),
                     size=(0.1, 0.1, 0.1),
                     orientation=Rotation.from_euler('xyz', [0, 0, t*math.pi/10]).as_quat(),
                     color=(0.5, 0.5, 0.5 + t/40),
                     label=f"box {t}",
                 )
                 for t in range(20)
             ],
         ),
    })
``` 
