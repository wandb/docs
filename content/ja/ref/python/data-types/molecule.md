---
title: Molecule
menu:
  reference:
    identifier: ja-ref-python-data-types-molecule
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/molecule.py#L25-L241 >}}

Wandb クラスは 3D 分子データ用です。

```python
Molecule(
    data_or_path: Union[str, 'TextIO'],
    caption: Optional[str] = None,
    **kwargs
) -> None
```

| Args |  |
| :--- | :--- |
|  `data_or_path` |  (string, io) Molecule はファイル名または io オブジェクトから初期化できます。 |
|  `caption` |  (string) 表示用の分子に関連付けられたキャプション。 |

## メソッド

### `from_rdkit`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/molecule.py#L99-L163)

```python
@classmethod
from_rdkit(
    data_or_path: "RDKitDataType",
    caption: Optional[str] = None,
    convert_to_3d_and_optimize: bool = (True),
    mmff_optimize_molecule_max_iterations: int = 200
) -> "Molecule"
```

RDKit がサポートするファイル/オブジェクトタイプを wandb.Molecule に変換します。

| Args |  |
| :--- | :--- |
|  `data_or_path` |  (string, rdkit.Chem.rdchem.Mol) Molecule はファイル名または rdkit.Chem.rdchem.Mol オブジェクトから初期化できます。 |
|  `caption` |  (string) 表示用の分子に関連付けられたキャプション。 |
|  `convert_to_3d_and_optimize` |  (bool) 3D 座標を持つ rdkit.Chem.rdchem.Mol に変換します。これは複雑な分子の場合、時間がかかるため、高価な操作です。 |
|  `mmff_optimize_molecule_max_iterations` |  (int) rdkit.Chem.AllChem.MMFFOptimizeMolecule で使用する反復回数 |

### `from_smiles`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/molecule.py#L165-L202)

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

SMILES 文字列を wandb.Molecule に変換します。

| Args |  |
| :--- | :--- |
|  `data` |  (string) SMILES 文字列。 |
|  `caption` |  (string) 表示用の分子に関連付けられたキャプション |
|  `sanitize` |  (bool) RDKit の定義により、分子が化学的に妥当かどうかをチェックします。 |
|  `convert_to_3d_and_optimize` |  (bool) 3D 座標で rdkit.Chem.rdchem.Mol に変換します。複雑な分子の場合、時間がかかるため、高価な操作です。 |
|  `mmff_optimize_molecule_max_iterations` |  (int) rdkit.Chem.AllChem.MMFFOptimizeMolecule で使用する反復回数 |

| クラス変数 |  |
| :--- | :--- |
|  `SUPPORTED_RDKIT_TYPES`<a id="SUPPORTED_RDKIT_TYPES"></a> |   |
|  `SUPPORTED_TYPES`<a id="SUPPORTED_TYPES"></a> |   |