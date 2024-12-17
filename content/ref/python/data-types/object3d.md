---
title: Object3D
---

{{< cta-button githubLink="https://www.github.com/wandb/wandb/tree/v0.18.7/wandb/sdk/data_types/object_3d.py#L190-L466" >}}

Wandb class for 3D point clouds.

```python
Object3D(
    data_or_path: Union['np.ndarray', str, 'TextIO', dict],
    **kwargs
) -> None
```

| Args |  |
| :--- | :--- |
|  `data_or_path` |  (numpy array, string, io) Object3D can be initialized from a file or a numpy array. You can pass a path to a file or an io object and a file_type which must be one of SUPPORTED_TYPES |

The shape of the numpy array must be one of either:

```
[[x y z],       ...] nx3
[[x y z c],     ...] nx4 where c is a category with supported range [1, 14]
[[x y z r g b], ...] nx6 where is rgb is color
```

## Methods

### `from_file`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.7/wandb/sdk/data_types/object_3d.py#L336-L353)

```python
@classmethod
from_file(
    data_or_path: Union['TextIO', str],
    file_type: Optional['FileFormat3D'] = None
) -> "Object3D"
```

Initializes Object3D from a file or stream.

| Args |  |
| :--- | :--- |
|  data_or_path (Union["TextIO", str]): A path to a file or a `TextIO` stream. file_type (str): Specifies the data format passed to `data_or_path`. Required when `data_or_path` is a `TextIO` stream. This parameter is ignored if a file path is provided. The type is taken from the file extension. |

### `from_numpy`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.7/wandb/sdk/data_types/object_3d.py#L355-L384)

```python
@classmethod
from_numpy(
    data: "np.ndarray"
) -> "Object3D"
```

Initializes Object3D from a numpy array.

| Args |  |
| :--- | :--- |
|  data (numpy array): Each entry in the array will represent one point in the point cloud. |

The shape of the numpy array must be one of either:

```
[[x y z],       ...]  # nx3.
[[x y z c],     ...]  # nx4 where c is a category with supported range [1, 14].
[[x y z r g b], ...]  # nx6 where is rgb is color.
```

### `from_point_cloud`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.7/wandb/sdk/data_types/object_3d.py#L386-L420)

```python
@classmethod
from_point_cloud(
    points: Sequence['Point'],
    boxes: Sequence['Box3D'],
    vectors: Optional[Sequence['Vector3D']] = None,
    point_cloud_type: "PointCloudType" = "lidar/beta"
) -> "Object3D"
```

Initializes Object3D from a python object.

| Args |  |
| :--- | :--- |
|  points (Sequence["Point"]): The points in the point cloud. boxes (Sequence["Box3D"]): 3D bounding boxes for labeling the point cloud. Boxes are displayed in point cloud visualizations. vectors (Optional[Sequence["Vector3D"]]): Each vector is displayed in the point cloud visualization. Can be used to indicate directionality of bounding boxes. Defaults to None. point_cloud_type ("lidar/beta"): At this time, only the "lidar/beta" type is supported. Defaults to "lidar/beta". |

| Class Variables |  |
| :--- | :--- |
|  `SUPPORTED_POINT_CLOUD_TYPES`<a id="SUPPORTED_POINT_CLOUD_TYPES"></a> |   |
|  `SUPPORTED_TYPES`<a id="SUPPORTED_TYPES"></a> |   |
