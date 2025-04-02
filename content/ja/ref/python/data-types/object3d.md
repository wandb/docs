---
title: Object3D
menu:
  reference:
    identifier: ja-ref-python-data-types-object3d
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/object_3d.py#L186-L462 >}}

3Dポイントクラウド用のWandbクラス。

```python
Object3D(
    data_or_path: Union['np.ndarray', str, 'TextIO', dict],
    **kwargs
) -> None
```

| arg |   |
| :--- | :--- |
| `data_or_path` | (numpy array, string, io) Object3D は、ファイルまたは numpy array から初期化できます。ファイルへのパスまたは io オブジェクトと、SUPPORTED_TYPES のいずれかである必要がある file_type を渡すことができます。 |

numpy array の形状は、次のいずれかである必要があります。

```
[[x y z],       ...] nx3
[[x y z c],     ...] nx4  ここで、c はサポートされている範囲 [1, 14] のカテゴリです。
[[x y z r g b], ...] nx6 ここで、rgb は色です。
```

## メソッド

### `from_file`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/object_3d.py#L332-L349)

```python
@classmethod
from_file(
    data_or_path: Union['TextIO', str],
    file_type: Optional['FileFormat3D'] = None
) -> "Object3D"
```

ファイルまたはストリームから Object3D を初期化します。

| arg |   |
| :--- | :--- |
| data_or_path (Union["TextIO", str]): ファイルへのパスまたは `TextIO` ストリーム。 file_type (str): `data_or_path` に渡されるデータ形式を指定します。`data_or_path` が `TextIO` ストリームの場合に必要です。このパラメータは、ファイルパスが指定されている場合は無視されます。タイプはファイル拡張子から取得されます。 |

### `from_numpy`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/object_3d.py#L351-L380)

```python
@classmethod
from_numpy(
    data: "np.ndarray"
) -> "Object3D"
```

numpy array から Object3D を初期化します。

| arg |   |
| :--- | :--- |
| data (numpy array): array の各エントリは、ポイントクラウド内の 1 つのポイントを表します。 |

numpy array の形状は、次のいずれかである必要があります。

```
[[x y z],       ...]  # nx3.
[[x y z c],     ...]  # nx4 ここで、c はサポートされている範囲 [1, 14] のカテゴリです。
[[x y z r g b], ...]  # nx6 ここで、rgb は色です。
```

### `from_point_cloud`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/object_3d.py#L382-L416)

```python
@classmethod
from_point_cloud(
    points: Sequence['Point'],
    boxes: Sequence['Box3D'],
    vectors: Optional[Sequence['Vector3D']] = None,
    point_cloud_type: "PointCloudType" = "lidar/beta"
) -> "Object3D"
```

python オブジェクトから Object3D を初期化します。

| arg |   |
| :--- | :--- |
| points (Sequence["Point"]): ポイントクラウド内のポイント。 boxes (Sequence["Box3D"]): ポイントクラウドにラベルを付けるための 3D バウンディングボックス。ボックスは、ポイントクラウドの可視化に表示されます。 vectors (Optional[Sequence["Vector3D"]]): 各ベクターは、ポイントクラウドの可視化に表示されます。バウンディングボックスの方向を示すために使用できます。デフォルトは None です。 point_cloud_type ("lidar/beta"): 現在、"lidar/beta" タイプのみがサポートされています。デフォルトは "lidar/beta" です。 |

| クラス変数 |   |
| :--- | :--- |
| `SUPPORTED_POINT_CLOUD_TYPES`<a id="SUPPORTED_POINT_CLOUD_TYPES"></a> |   |
| `SUPPORTED_TYPES`<a id="SUPPORTED_TYPES"></a> |   |
