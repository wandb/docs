# Object3D

[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)GitHubでソースを表示](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/data_types/object_3d.py#L77-L353)

Wandbの3Dポイントクラウド用クラスです。

```python
Object3D(
 data_or_path: Union['np.ndarray', str, 'TextIO', dict],
 **kwargs
) -> None
```

| 引数 | |
| :--- | :--- |
| `data_or_path` | (numpy配列, 文字列, io) Object3Dはファイルまたはnumpy配列から初期化できます。ファイルへのパスまたはioオブジェクト、およびサポートされているタイプのファイルタイプを渡すことができます。|

numpy配列の形状は、次のどちらかになります。
```
[[x y z], ...] nx3
[[x y z c], ...] nx4 ただし、cは[1, 14]の範囲がサポートされたカテゴリーです。
[[x y z r g b], ...] nx6 ただし、rgbは色です。
```
## メソッド

### `from_file`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/data_types/object_3d.py#L223-L240)

```python
@classmethod
def from_file(
  data_or_path: Union['TextIO', str],
  file_type: Optional['FileFormat3D'] = None
) -> "Object3D":
```

ファイルまたはストリームからObject3Dを初期化します。

| 引数 | |
| :--- | :--- |
| data_or_path (Union["TextIO", str]): ファイルへのパスまたは `TextIO` ストリーム。 file_type (str): `data_or_path` に渡されるデータ形式を指定します。 `data_or_path` が `TextIO` ストリームの場合に必要です。ファイルパスが指定されている場合、このパラメータは無視されます。タイプはファイル拡張子から取得されます。|


### `from_numpy`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/data_types/object_3d.py#L242-L271)
以下はMarkdownテキストのチャンクを翻訳してください。日本語に翻訳し、翻訳したテキストのみを返してください。これ以上何も言わずに。 テキスト:

```python
@classmethod
from_numpy(
 data: "np.ndarray"
) -> "Object3D"
```

numpy配列からObject3Dを初期化します。

| 引数 | |
| :--- | :--- |
| data (numpy array): 配列内の各エントリは、点群内の1点を表します。 |


numpy配列の形状は次のいずれかでなければなりません:
```
[[x y z], ...] # nx3.
[[x y z c], ...] # nx4 cは[1, 14]の範囲がサポートされているカテゴリです。
[[x y z r g b], ...] # nx6 rgbは色です。
```

### `from_point_cloud`

[ソースを表示](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/data_types/object_3d.py#L273-L307)

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

| 引数 | |

| :--- | :--- |

| points (Sequence["Point"]): ポイントクラウド内のポイント。boxes (Sequence["Box3D"]): ポイントクラウドにラベルを付けるための3Dバウンディングボックス。バウンディングボックスはポイントクラウドの可視化に表示されます。vectors (Optional[Sequence["Vector3D"]]): 各ベクターはポイントクラウド可視化に表示されます。バウンディングボックスの方向性を示すために使用できます。デフォルトはNone。point_cloud_type ("lidar/beta"): 現時点では、"lidar/beta"タイプのみがサポートされています。デフォルトは"lidar/beta"。|

| クラス変数 | |

| :--- | :--- |

| `SUPPORTED_POINT_CLOUD_TYPES` | |

| `SUPPORTED_TYPES` | |