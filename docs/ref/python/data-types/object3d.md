# Object3D



[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)View source on GitHub](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/data_types/object_3d.py#L77-L353)



Wandb class for 3D point clouds.

```python
Object3D(
 data_or_path: Union['np.ndarray', str, 'TextIO', dict],
 **kwargs
) -> None
```





| Arguments | |
| :--- | :--- |
| `data_or_path` | (numpy array, string, io) Object3D can be initialized from a file or a numpy array. You can pass a path to a file or an io object and a file_type which must be one of SUPPORTED_TYPES |


The shape of the numpy array must be one of either:
```
[[x y z], ...] nx3
[[x y z c], ...] nx4 where c is a category with supported range [1, 14]
[[x y z r g b], ...] nx6 where is rgb is color
```

## Methods

### `from_file`



[View source](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/data_types/object_3d.py#L223-L240)

```python
@classmethod
from_file(
 data_or_path: Union['TextIO', str],
 file_type: Optional['FileFormat3D'] = None
) -> "Object3D"
```

Initializes Object3D from a file or stream.


| Arguments | |
| :--- | :--- |
| data_or_path (Union["TextIO", str]): A path to a file or a `TextIO` stream. file_type (str): Specifies the data format passed to `data_or_path`. Required when `data_or_path` is a `TextIO` stream. This parameter is ignored if a file path is provided. The type is taken from the file extension. |



### `from_numpy`



[View source](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/data_types/object_3d.py#L242-L271)

```python
@classmethod
from_numpy(
 data: "np.ndarray"
) -> "Object3D"
```

Initializes Object3D from a numpy array.


| Arguments | |
| :--- | :--- |
| data (numpy array): Each entry in the array will represent one point in the point cloud. |


The shape of the numpy array must be one of either:
```
[[x y z], ...] # nx3.
[[x y z c], ...] # nx4 where c is a category with supported range [1, 14].
[[x y z r g b], ...] # nx6 where is rgb is color.
```

### `from_point_cloud`



[View source](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/data_types/object_3d.py#L273-L307)

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


| Arguments | |
| :--- | :--- |
| points (Sequence["Point"]): The points in the point cloud. boxes (Sequence["Box3D"]): 3D bounding boxes for labeling the point cloud. Boxes are displayed in point cloud visualizations. vectors (Optional[Sequence["Vector3D"]]): Each vector is displayed in the point cloud visualization. Can be used to indicate directionality of bounding boxes. Defaults to None. point_cloud_type ("lidar/beta"): At this time, only the "lidar/beta" type is supported. Defaults to "lidar/beta". |







| Class Variables | |
| :--- | :--- |
| `SUPPORTED_POINT_CLOUD_TYPES` | |
| `SUPPORTED_TYPES` | |

