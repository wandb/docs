---
title: Object3D
menu:
  reference:
    identifier: ja-ref-python-data-types-object3d
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/object_3d.py#L186-L462 >}}

3DポイントクラウドのためのWandbクラス。

```python
Object3D(
    data_or_path: Union['np.ndarray', str, 'TextIO', dict],
    **kwargs
) -> None
```

| 引数 |  |
| :--- | :--- |
| `data_or_path` |  (numpy array, string, io) Object3Dはファイルまたはnumpy配列から初期化できます。ファイルへのパスまたはio オブジェクトと `SUPPORTED_TYPES` のいずれかである必要がある `file_type` を渡すことができます。|

numpy 配列の形状は次のいずれかでなければなりません：

```
[[x y z],       ...] nx3
[[x y z c],     ...] nx4 ここで c は[1, 14] の範囲内のカテゴリです
[[x y z r g b], ...] nx6 ここで rgb は色です
```

## メソッド

### `from_file`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/object_3d.py#L332-L349)

```python
@classmethod
from_file(
    data_or_path: Union['TextIO', str],
    file_type: Optional['FileFormat3D'] = None
) -> "Object3D"
```

ファイルまたはストリームから Object3D を初期化します。

| 引数 |  |
| :--- | :--- |
| data_or_path (Union["TextIO", str]): ファイルへのパスまたは `TextIO` ストリーム。file_type (str): `data_or_path` に渡されるデータ形式を指定します。 `data_or_path` が `TextIO` ストリームである場合は必須です。ファイルパスが提供されている場合はこのパラメータは無視されます。タイプはファイル拡張子から取得されます。|

### `from_numpy`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/object_3d.py#L351-L380)

```python
@classmethod
from_numpy(
    data: "np.ndarray"
) -> "Object3D"
```

numpy 配列から Object3D を初期化します。

| 引数 |  |
| :--- | :--- |
| data (numpy array): 配列の各エントリはポイントクラウドの1ポイントを表します。 |

numpy 配列の形状は次のいずれかでなければなりません：

```
[[x y z],       ...]  # nx3.
[[x y z c],     ...]  # nx4 ここで c は [1, 14] の範囲内のカテゴリです。
[[x y z r g b], ...]  # nx6 ここで rgb は色です。
```

### `from_point_cloud`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/object_3d.py#L382-L416)

```python
@classmethod
from_point_cloud(
    points: Sequence['Point'],
    boxes: Sequence['Box3D'],
    vectors: Optional[Sequence['Vector3D']] = None,
    point_cloud_type: "PointCloudType" = "lidar/beta"
) -> "Object3D"
```

Python オブジェクトから Object3D を初期化します。

| 引数 |  |
| :--- | :--- |
| points (Sequence["Point"]): ポイントクラウドの点。boxes (Sequence["Box3D"]): ポイントクラウドのラベル付け用3Dバウンディングボックス。ボックスはポイントクラウドの可視化で表示されます。vectors (Optional[Sequence["Vector3D"]]): 各ベクトルはポイントクラウドの可視化で表示されます。バウンディングボックスの方向性を示すために使用できます。デフォルトは None です。point_cloud_type ("lidar/beta"): 現時点では「lidar/beta」タイプのみサポートしています。デフォルトは「lidar/beta」です。|

| クラス変数 |  |
| :--- | :--- |
| `SUPPORTED_POINT_CLOUD_TYPES`<a id="SUPPORTED_POINT_CLOUD_TYPES"></a> |   |
| `SUPPORTED_TYPES`<a id="SUPPORTED_TYPES"></a> |   |