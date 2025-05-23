---
title: Molecule
menu:
  reference:
    identifier: ko-ref-python-data-types-molecule
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/molecule.py#L25-L241 >}}

3D 분자 데이터를 위한 Wandb 클래스입니다.

```python
Molecule(
    data_or_path: Union[str, 'TextIO'],
    caption: Optional[str] = None,
    **kwargs
) -> None
```

| ARG |  |
| :--- | :--- |
|  `data_or_path` |  (string, io) Molecule은 파일 이름이나 io 오브젝트에서 초기화할 수 있습니다. |
|  `caption` |  (string) 표시를 위해 분자와 연결된 캡션입니다. |

## Methods

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

RDKit에서 지원하는 파일/오브젝트 유형을 wandb.Molecule로 변환합니다.

| ARG |  |
| :--- | :--- |
|  `data_or_path` |  (string, rdkit.Chem.rdchem.Mol) Molecule은 파일 이름이나 rdkit.Chem.rdchem.Mol 오브젝트에서 초기화할 수 있습니다. |
|  `caption` |  (string) 표시를 위해 분자와 연결된 캡션입니다. |
|  `convert_to_3d_and_optimize` |  (bool) 3D 좌표로 rdkit.Chem.rdchem.Mol로 변환합니다. 이는 복잡한 분자에 대해 시간이 오래 걸릴 수 있는 비용이 많이 드는 작업입니다. |
|  `mmff_optimize_molecule_max_iterations` |  (int) rdkit.Chem.AllChem.MMFFOptimizeMolecule에서 사용할 반복 횟수입니다. |

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

SMILES 문자열을 wandb.Molecule로 변환합니다.

| ARG |  |
| :--- | :--- |
|  `data` |  (string) SMILES 문자열. |
|  `caption` |  (string) 표시를 위해 분자와 연결된 캡션 |
|  `sanitize` |  (bool) 분자가 RDKit의 정의에 따라 화학적으로 합리적인지 확인합니다. |
|  `convert_to_3d_and_optimize` |  (bool) 3D 좌표로 rdkit.Chem.rdchem.Mol로 변환합니다. 이는 복잡한 분자에 대해 시간이 오래 걸릴 수 있는 비용이 많이 드는 작업입니다. |
|  `mmff_optimize_molecule_max_iterations` |  (int) rdkit.Chem.AllChem.MMFFOptimizeMolecule에서 사용할 반복 횟수입니다. |

| Class Variables |  |
| :--- | :--- |
|  `SUPPORTED_RDKIT_TYPES`<a id="SUPPORTED_RDKIT_TYPES"></a> |   |
|  `SUPPORTED_TYPES`<a id="SUPPORTED_TYPES"></a> |   |
