import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx'

# Object3D

<CTAButtons githubLink='https://github.com/wandb/wandb/blob/main/wandb/sdk/data_types/object_3d.py'/>




## <kbd>class</kbd> `Object3D`
Wandb class for 3D point clouds. 



**Args:**
 
 - `data_or_path`:  (numpy array, string, io)  Object3D can be initialized from a file or a numpy array. 

 You can pass a path to a file or an io object and a file_type  which must be one of SUPPORTED_TYPES 

The shape of the numpy array must be one of either: ```
[[x y z],       ...] nx3
[[x y z c],     ...] nx4 where c is a category with supported range [1, 14]
[[x y z r g b], ...] nx6 where is rgb is color
``` 

### <kbd>method</kbd> `Object3D.__init__`

```python
__init__(
    data_or_path: Union[ForwardRef('np.ndarray'), str, ForwardRef('TextIO'), dict],
    **kwargs: Optional[str, ForwardRef('FileFormat3D')]
) → None
```








---

### <kbd>classmethod</kbd> `Object3D.from_file`

```python
from_file(
    data_or_path: Union[ForwardRef('TextIO'), str],
    file_type: Optional[ForwardRef('FileFormat3D')] = None
) → Object3D
```

Initializes Object3D from a file or stream. 



**Args:**
 
 - `data_or_path` (Union["TextIO", str]):  A path to a file or a `TextIO` stream. 
 - `file_type` (str):  Specifies the data format passed to `data_or_path`. Required when `data_or_path` is a  `TextIO` stream. This parameter is ignored if a file path is provided. The type is taken from the file extension. 

---

### <kbd>classmethod</kbd> `Object3D.from_numpy`

```python
from_numpy(data: 'np.ndarray') → Object3D
```

Initializes Object3D from a numpy array. 



**Args:**
 
 - `data` (numpy array):  Each entry in the array will  represent one point in the point cloud. 



The shape of the numpy array must be one of either: ```
[[x y z],       ...]  # nx3.
[[x y z c],     ...]  # nx4 where c is a category with supported range [1, 14].
[[x y z r g b], ...]  # nx6 where is rgb is color.
``` 

---

### <kbd>classmethod</kbd> `Object3D.from_point_cloud`

```python
from_point_cloud(
    points: Sequence[ForwardRef('Point')],
    boxes: Sequence[ForwardRef('Box3D')],
    vectors: Optional[Sequence[ForwardRef('Vector3D')]] = None,
    point_cloud_type: 'PointCloudType' = 'lidar/beta'
) → Object3D
```

Initializes Object3D from a python object. 



**Args:**
 
 - `points` (Sequence["Point"]):  The points in the point cloud. 
 - `boxes` (Sequence["Box3D"]):  3D bounding boxes for labeling the point cloud. Boxes are displayed in point cloud visualizations. 
 - `vectors` (Optional[Sequence["Vector3D"]]):  Each vector is displayed in the point cloud  visualization. Can be used to indicate directionality of bounding boxes. Defaults to None. 
 - `point_cloud_type` ("lidar/beta"):  At this time, only the "lidar/beta" type is supported. Defaults to "lidar/beta". 

---

### <kbd>classmethod</kbd> `Object3D.get_media_subdir`

```python
get_media_subdir() → str
```





---

### <kbd>classmethod</kbd> `Object3D.seq_to_json`

```python
seq_to_json(
    seq: Sequence[ForwardRef('BatchableMedia')],
    run: 'LocalRun',
    key: str,
    step: Union[int, str]
) → dict
```





---

### <kbd>method</kbd> `Object3D.to_json`

```python
to_json(
    run_or_artifact: Union[ForwardRef('LocalRun'), ForwardRef('Artifact')]
) → dict
```