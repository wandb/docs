# 分子

[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)GitHubでソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/data_types/molecule.py#L23-L239)

Wandbの3D分子データ用クラス。

```python
Molecule(
 data_or_path: Union[str, 'TextIO'],
 caption: Optional[str] = None,
 **kwargs
) -> None
```

| 引数 | |
| :--- | :--- |
| `data_or_path` | (string, io) 分子はファイル名またはioオブジェクトから初期化できます。 |
| `caption` | (string) 分子に関連付けられた表示用のキャプション。 |

## メソッド
### `from_rdkit`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/data_types/molecule.py#L97-L161)

```python
@classmethod
from_rdkit(
 data_or_path: "RDKitDataType",
 caption: Optional[str] = None,
 convert_to_3d_and_optimize: bool = (True),
 mmff_optimize_molecule_max_iterations: int = 200
) -> "Molecule"
```

RDKit対応のファイル/オブジェクトタイプをwandb.Moleculeに変換します。

| 引数 | 説明 |
| :--- | :--- |
| `data_or_path` | (string, rdkit.Chem.rdchem.Mol) モLECULeはファイル名またはrdkit.Chem.rdchem.Molオブジェクトから初期化できます。 |
| `caption` | (string) 分子に関連するキャプションを表示します。 |
| `convert_to_3d_and_optimize` | (bool) rdkit.Chem.rdchem.Molを3D座標で変換します。これは複雑な分子の場合、長時間かかる可能性があるコストのかかる操作です。 |
| `mmff_optimize_molecule_max_iterations` | (int) rdkit.Chem.AllChem.MMFFOptimizeMoleculeで使用する反復回数 |



### `from_smiles`
ソースを表示（[View source](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/data_types/molecule.py#L163-L200)）

```python
@classmethod
from_smiles(
 data: str,
 caption: Optional[str] = None,
 sanitize: bool = (True),
 convert_to_3d_and_optimize: bool = (True),
 mmff_optimize_molecule_max_iterations: int = 200
) -> "Molecule"
```

SMILES文字列をwandb.Moleculeに変換します。

| 引数 | |
| :--- | :--- |
| `data` | (string) SMILES文字列。 |
| `caption` | (string) 分子表示に関連付けられたキャプション。 |
| `sanitize` | (bool) RDKitの定義に基づいて、分子が化学的に適切であるかどうかを確認します。 |
| `convert_to_3d_and_optimize` | (bool) rdkit.Chem.rdchem.Molに3D座標をもつものに変換します。これは複雑な分子の場合、処理が長時間かかることがある高コストな操作です。 |
| `mmff_optimize_molecule_max_iterations` | (int) rdkit.Chem.AllChem.MMFFOptimizeMoleculeで使用する反復回数。
| クラス変数 | |

| :--- | :--- |

| `SUPPORTED_RDKIT_TYPES` | |

| `SUPPORTED_TYPES` | |