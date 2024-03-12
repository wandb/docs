
# 분자

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/sdk/data_types/molecule.py#L25-L241' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>GitHub에서 소스 보기</a></button></p>


3D 분자 데이터를 위한 Wandb 클래스입니다.

```python
Molecule(
    data_or_path: Union[str, 'TextIO'],
    caption: Optional[str] = None,
    **kwargs
) -> None
```

| 인수 |  |
| :--- | :--- |
|  `data_or_path` |  (문자열, io) 파일 이름이나 io 오브젝트로 분자를 초기화할 수 있습니다. |
|  `caption` |  (문자열) 분자를 표시할 때 연결된 캡션입니다. |

## 메소드

### `from_rdkit`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/sdk/data_types/molecule.py#L99-L163)

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
|  `data_or_path` |  (문자열, rdkit.Chem.rdchem.Mol) 파일 이름이나 rdkit.Chem.rdchem.Mol 오브젝트로 분자를 초기화할 수 있습니다. |
|  `caption` |  (문자열) 분자를 표시할 때 연결된 캡션입니다. |
|  `convert_to_3d_and_optimize` |  (불리언) 3D 좌표를 가진 rdkit.Chem.rdchem.Mol로 변환합니다. 이는 복잡한 분자에 대해서는 많은 시간이 걸릴 수 있는 비용이 많이 드는 작업입니다. |
|  `mmff_optimize_molecule_max_iterations` |  (정수) rdkit.Chem.AllChem.MMFFOptimizeMolecule에서 사용할 반복 횟수입니다. |

### `from_smiles`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/sdk/data_types/molecule.py#L165-L202)

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
|  `data` |  (문자열) SMILES 문자열입니다. |
|  `caption` |  (문자열) 분자를 표시할 때 연결된 캡션입니다. |
|  `sanitize` |  (불리언) RDKit의 정의에 따라 분자가 화학적으로 합리적인지 확인합니다. |
|  `convert_to_3d_and_optimize` |  (불리언) 3D 좌표를 가진 rdkit.Chem.rdchem.Mol로 변환합니다. 이는 복잡한 분자에 대해서는 많은 시간이 걸릴 수 있는 비용이 많이 드는 작업입니다. |
|  `mmff_optimize_molecule_max_iterations` |  (정수) rdkit.Chem.AllChem.MMFFOptimizeMolecule에서 사용할 반복 횟수입니다. |

| 클래스 변수 |  |
| :--- | :--- |
|  `SUPPORTED_RDKIT_TYPES`<a id="SUPPORTED_RDKIT_TYPES"></a> |   |
|  `SUPPORTED_TYPES`<a id="SUPPORTED_TYPES"></a> |   |