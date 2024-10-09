# Molecule

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/data_types/molecule.py#L25-L241' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

3D 분자 데이터에 대한 Wandb 클래스입니다.

```python
Molecule(
    data_or_path: Union[str, 'TextIO'],
    caption: Optional[str] = None,
    **kwargs
) -> None
```

| 인수 |  |
| :--- | :--- |
|  `data_or_path` |  (문자열, io) Molecule은 파일 이름이나 io 오브젝트에서 초기화될 수 있습니다. |
|  `caption` |  (문자열) 디스플레이를 위한 분자에 연관된 캡션입니다. |

## 메소드

### `from_rdkit`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/data_types/molecule.py#L99-L163)

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

| 인수 |  |
| :--- | :--- |
|  `data_or_path` |  (문자열, rdkit.Chem.rdchem.Mol) Molecule은 파일 이름이나 rdkit.Chem.rdchem.Mol 오브젝트에서 초기화될 수 있습니다. |
|  `caption` |  (문자열) 디스플레이를 위한 분자에 연관된 캡션입니다. |
|  `convert_to_3d_and_optimize` |  (bool) 3D 좌표로 rdkit.Chem.rdchem.Mol로 변환합니다. 이는 복잡한 분자의 경우 오래 걸릴 수 있는 고비용 작업입니다. |
|  `mmff_optimize_molecule_max_iterations` |  (int) rdkit.Chem.AllChem.MMFFOptimizeMolecule에서 사용할 반복 횟수 |

### `from_smiles`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/data_types/molecule.py#L165-L202)

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

| 인수 |  |
| :--- | :--- |
|  `data` |  (문자열) SMILES 문자열. |
|  `caption` |  (문자열) 디스플레이를 위한 분자에 연관된 캡션 |
|  `sanitize` |  (bool) RDKit의 정의에 따라 화학적으로 타당한지 여부를 확인합니다. |
|  `convert_to_3d_and_optimize` |  (bool) 3D 좌표로 rdkit.Chem.rdchem.Mol로 변환합니다. 이는 복잡한 분자의 경우 오래 걸릴 수 있는 고비용 작업입니다. |
|  `mmff_optimize_molecule_max_iterations` |  (int) rdkit.Chem.AllChem.MMFFOptimizeMolecule에서 사용할 반복 횟수 |

| 클래스 변수 |  |
| :--- | :--- |
|  `SUPPORTED_RDKIT_TYPES`<a id="SUPPORTED_RDKIT_TYPES"></a> |   |
|  `SUPPORTED_TYPES`<a id="SUPPORTED_TYPES"></a> |   |