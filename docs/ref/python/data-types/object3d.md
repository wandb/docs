# Object3D

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/data_types/object_3d.py#L79-L355' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

Wandbクラスは3Dポイントクラウドのためのものです。

```python
Object3D(
    data_or_path: Union['np.ndarray', str, 'TextIO', dict],
    **kwargs
) -> None
```

| 引数 |  |
| :--- | :--- |
|  `data_or_path` |  (numpy array, string, io) Object3Dはファイルまたはnumpy配列から初期化できます。ファイルへのパス、またはioオブジェクトと、SUPPORTED_TYPESのいずれかである必要があるfile_typeを渡すことができます。 |

numpy配列の形状は以下のいずれかでなければなりません:

```
[[x y z],       ...] nx3
[[x y z c],     ...] nx4 ここでcはサポートされている範囲[1, 14]のカテゴリです
[[x y z r g b], ...] nx6 ここでrgbは色です
```

## メソッド

### `from_file`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/data_types/object_3d.py#L225-L242)

```python
@classmethod
from_file(
    data_or_path: Union['TextIO', str],
    file_type: Optional['FileFormat3D'] = None
) -> "Object3D"
```

Object3Dをファイルまたはストリームから初期化します。

| 引数 |  |
| :--- | :--- |
|  data_or_path (Union["TextIO", str]): ファイルへのパスまたは `TextIO` ストリーム。 file_type (str): `data_or_path`に渡されるデータ形式を指定します。 `data_or_path`が`TextIO`ストリームの場合に必要です。 ファイルパスが指定されている場合、このパラメータは無視されます。種類はファイル拡張子から取得されます。 |

### `from_numpy`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/data_types/object_3d.py#L244-L273)

```python
@classmethod
from_numpy(
    data: "np.ndarray"
) -> "Object3D"
```

Object3Dをnumpy配列から初期化します。

| 引数 |  |
| :--- | :--- |
|  data (numpy array): 配列の各エントリがポイントクラウド内の1ポイントを表します。 |

numpy配列の形状は以下のいずれかでなければなりません:

```
[[x y z],       ...]  # nx3.
[[x y z c],     ...]  # nx4 ここでcはサポートされている範囲[1, 14]のカテゴリです。
[[x y z r g b], ...]  # nx6 ここでrgbは色です。
```

### `from_point_cloud`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/data_types/object_3d.py#L275-L309)

```python
@classmethod
from_point_cloud(
    points: Sequence['Point'],
    boxes: Sequence['Box3D'],
    vectors: Optional[Sequence['Vector3D']] = None,
    point_cloud_type: "PointCloudType" = "lidar/beta"
) -> "Object3D"
```

Object3DをPythonオブジェクトから初期化します。

| 引数 |  |
| :--- | :--- |
|  points (Sequence["Point"]): ポイントクラウドのポイント。 boxes (Sequence["Box3D"]): ポイントクラウドのラベル付けのための3Dバウンディングボックス。 ボックスはポイントクラウドの可視化に表示されます。 vectors (Optional[Sequence["Vector3D"]]): 各ベクターはポイントクラウドの可視化で表示されます。 ボックスの方向性を示すために使用できます。デフォルトはNone。 point_cloud_type ("lidar/beta"): 現在サポートされているのは "lidar/beta" タイプのみです。デフォルトは "lidar/beta"。 |

| クラス変数 |  |
| :--- | :--- |
|  `SUPPORTED_POINT_CLOUD_TYPES`<a id="SUPPORTED_POINT_CLOUD_TYPES"></a> |   |
|  `SUPPORTED_TYPES`<a id="SUPPORTED_TYPES"></a> |   |