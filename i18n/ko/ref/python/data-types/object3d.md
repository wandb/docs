
# Object3D

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/sdk/data_types/object_3d.py#L79-L355' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>GitHub에서 소스 보기</a></button></p>


3D 포인트 클라우드를 위한 Wandb 클래스입니다.

```python
Object3D(
    data_or_path: Union['np.ndarray', str, 'TextIO', dict],
    **kwargs
) -> None
```

| 인수 |  |
| :--- | :--- |
|  `data_or_path` |  (numpy 배열, 문자열, io) Object3D는 파일 또는 numpy 배열에서 초기화될 수 있습니다. 파일 경로 또는 io 오브젝트를 전달하고 file_type은 SUPPORTED_TYPES 중 하나여야 합니다. |

numpy 배열의 형태는 다음 중 하나여야 합니다:

```
[[x y z],       ...] nx3
[[x y z c],     ...] nx4 여기서 c는 지원 범위가 [1, 14]인 카테고리입니다.
[[x y z r g b], ...] nx6 여기서 rgb는 색상입니다.
```

## 메소드

### `from_file`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/sdk/data_types/object_3d.py#L225-L242)

```python
@classmethod
from_file(
    data_or_path: Union['TextIO', str],
    file_type: Optional['FileFormat3D'] = None
) -> "Object3D"
```

파일 또는 스트림에서 Object3D를 초기화합니다.

| 인수 |  |
| :--- | :--- |
|  data_or_path (Union["TextIO", str]): 파일 경로 또는 `TextIO` 스트림. file_type (str): `data_or_path`로 전달된 데이터 형식을 지정합니다. `data_or_path`가 `TextIO` 스트림일 때 필요합니다. 파일 경로가 제공되면 이 파라미터는 무시됩니다. 유형은 파일 확장자에서 가져옵니다. |

### `from_numpy`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/sdk/data_types/object_3d.py#L244-L273)

```python
@classmethod
from_numpy(
    data: "np.ndarray"
) -> "Object3D"
```

numpy 배열에서 Object3D를 초기화합니다.

| 인수 |  |
| :--- | :--- |
|  data (numpy 배열): 배열의 각 항목은 포인트 클라우드의 한 점을 나타냅니다. |

numpy 배열의 형태는 다음 중 하나여야 합니다:

```
[[x y z],       ...]  # nx3.
[[x y z c],     ...]  # nx4 여기서 c는 지원 범위가 [1, 14]인 카테고리입니다.
[[x y z r g b], ...]  # nx6 여기서 rgb는 색상입니다.
```

### `from_point_cloud`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/sdk/data_types/object_3d.py#L275-L309)

```python
@classmethod
from_point_cloud(
    points: Sequence['Point'],
    boxes: Sequence['Box3D'],
    vectors: Optional[Sequence['Vector3D']] = None,
    point_cloud_type: "PointCloudType" = "lidar/beta"
) -> "Object3D"
```

파이썬 오브젝트에서 Object3D를 초기화합니다.

| 인수 |  |
| :--- | :--- |
|  points (Sequence["Point"]): 포인트 클라우드의 포인트들. boxes (Sequence["Box3D"]): 포인트 클라우드 라벨링을 위한 3D 바운딩 박스. 박스는 포인트 클라우드 시각화에 표시됩니다. vectors (Optional[Sequence["Vector3D"]]): 각 벡터는 포인트 클라우드 시각화에 표시됩니다. 바운딩 박스의 방향성을 나타내는 데 사용할 수 있습니다. 기본값은 None입니다. point_cloud_type ("lidar/beta"): 현재 "lidar/beta" 타입만 지원됩니다. 기본값은 "lidar/beta"입니다. |

| 클래스 변수 |  |
| :--- | :--- |
|  `SUPPORTED_POINT_CLOUD_TYPES`<a id="SUPPORTED_POINT_CLOUD_TYPES"></a> |   |
|  `SUPPORTED_TYPES`<a id="SUPPORTED_TYPES"></a> |   |