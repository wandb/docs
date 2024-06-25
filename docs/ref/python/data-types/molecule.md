
# Molecule

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/data_types/molecule.py#L25-L241' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

Wandbクラスは3D分子データ用です。

```python
Molecule(
    data_or_path: Union[str, 'TextIO'],
    caption: Optional[str] = None,
    **kwargs
) -> None
```

| 引数 |  |
| :--- | :--- |
|  `data_or_path` |  (string, io) ファイル名またはioオブジェクトから分子を初期化することができます。 |
|  `caption` |  (string) 表示用のキャプションを分子に関連付けます。 |

## メソッド

### `from_rdkit`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/data_types/molecule.py#L99-L163)

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

| 引数 |  |
| :--- | :--- |
|  `data_or_path` |  (string, rdkit.Chem.rdchem.Mol) ファイル名またはrdkit.Chem.rdchem.Molオブジェクトから分子を初期化することができます。 |
|  `caption` |  (string) 表示用のキャプションを分子に関連付けます。 |
|  `convert_to_3d_and_optimize` |  (bool) 3D座標でrdkit.Chem.rdchem.Molに変換します。これは複雑な分子には時間がかかる高価な操作です。 |
|  `mmff_optimize_molecule_max_iterations` |  (int) rdkit.Chem.AllChem.MMFFOptimizeMoleculeの使用回数。 |

### `from_smiles`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/data_types/molecule.py#L165-L202)

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

| 引数 |  |
| :--- | :--- |
|  `data` |  (string) SMILES文字列。 |
|  `caption` |  (string) 表示用のキャプションを分子に関連付けます。 |
|  `sanitize` |  (bool) RDKitの定義によって分子が化学的に妥当かどうか確認します。 |
|  `convert_to_3d_and_optimize` |  (bool) 3D座標でrdkit.Chem.rdchem.Molに変換します。これは複雑な分子には時間がかかる高価な操作です。 |
|  `mmff_optimize_molecule_max_iterations` |  (int) rdkit.Chem.AllChem.MMFFOptimizeMoleculeの使用回数。 |

| クラス変数 |  |
| :--- | :--- |
|  `SUPPORTED_RDKIT_TYPES`<a id="SUPPORTED_RDKIT_TYPES"></a> |   |
|  `SUPPORTED_TYPES`<a id="SUPPORTED_TYPES"></a> |   |