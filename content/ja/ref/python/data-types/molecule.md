---
title: Molecule
menu:
  reference:
    identifier: ja-ref-python-data-types-molecule
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/molecule.py#L25-L241 >}}

3D分子データ用のWandbクラス。

```python
Molecule(
    data_or_path: Union[str, 'TextIO'],
    caption: Optional[str] = None,
    **kwargs
) -> None
```

| arg |   |
| :--- | :--- |
| `data_or_path` | (string, io) Moleculeは、ファイル名またはio オブジェクトから初期化できます。 |
| `caption` | (string) 表示用に分子に関連付けられたキャプション。 |

## メソッド

### `from_rdkit`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/molecule.py#L99-L163)

```python
@classmethod
from_rdkit(
    data_or_path: "RDKitDataType",
    caption: Optional[str] = None,
    convert_to_3d_and_optimize: bool = (True),
    mmff_optimize_molecule_max_iterations: int = 200
) -> "Molecule"
```

RDKitがサポートするファイル/オブジェクトタイプをwandb.Moleculeに変換します。

| arg |   |
| :--- | :--- |
| `data_or_path` | (string, rdkit.Chem.rdchem.Mol) Moleculeは、ファイル名またはrdkit.Chem.rdchem.Molオブジェクトから初期化できます。 |
| `caption` | (string) 表示用に分子に関連付けられたキャプション。 |
| `convert_to_3d_and_optimize` | (bool) 3D座標を持つrdkit.Chem.rdchem.Molに変換します。これは、複雑な分子の場合、時間がかかる可能性のあるコストのかかる操作です。 |
| `mmff_optimize_molecule_max_iterations` | (int) rdkit.Chem.AllChem.MMFFOptimizeMolecule で使用する反復回数 |

### `from_smiles`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/molecule.py#L165-L202)

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

| arg |   |
| :--- | :--- |
| `data` | (string) SMILES文字列。 |
| `caption` | (string) 分子表示に関連付けられたキャプション |
| `sanitize` | (bool) RDKitの定義により、分子が化学的に合理的であるかどうかを確認します。 |
| `convert_to_3d_and_optimize` | (bool) 3D座標を持つrdkit.Chem.rdchem.Molに変換します。これは、複雑な分子の場合、時間がかかる可能性のあるコストのかかる操作です。 |
| `mmff_optimize_molecule_max_iterations` | (int) rdkit.Chem.AllChem.MMFFOptimizeMolecule で使用する反復回数 |

| クラス変数 |   |
| :--- | :--- |
| `SUPPORTED_RDKIT_TYPES`<a id="SUPPORTED_RDKIT_TYPES"></a> |   |
| `SUPPORTED_TYPES`<a id="SUPPORTED_TYPES"></a> |   |
