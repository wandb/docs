---
title: Molecule
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/sdk/data_types/molecule.py#L25-L250 >}}

W&B class for 3D Molecular data.

## Methods

### `from_rdkit`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/sdk/data_types/molecule.py#L98-L162)

```python
@classmethod
from_rdkit(
    data_or_path: "RDKitDataType",
    caption: Optional[str] = None,
    convert_to_3d_and_optimize: bool = (True),
    mmff_optimize_molecule_max_iterations: int = 200
) -> "Molecule"
```

Convert RDKit-supported file/object types to wandb.Molecule.

| Args |  |
| :--- | :--- |
|  `data_or_path` |  (string, rdkit.Chem.rdchem.Mol) Molecule can be initialized from a file name or an rdkit.Chem.rdchem.Mol object. |
|  `caption` |  (string) Caption associated with the molecule for display. |
|  `convert_to_3d_and_optimize` |  (bool) Convert to rdkit.Chem.rdchem.Mol with 3D coordinates. This is an expensive operation that may take a long time for complicated molecules. |
|  `mmff_optimize_molecule_max_iterations` |  (int) Number of iterations to use in rdkit.Chem.AllChem.MMFFOptimizeMolecule |

<!-- lazydoc-ignore-classmethod: internal -->


### `from_smiles`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/sdk/data_types/molecule.py#L164-L201)

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

Convert SMILES string to wandb.Molecule.

| Args |  |
| :--- | :--- |
|  `data` |  SMILES string. |
|  `caption` |  Caption associated with the molecule for display. |
|  `sanitize` |  Check if the molecule is chemically reasonable by the RDKit's definition. |
|  `convert_to_3d_and_optimize` |  Convert to rdkit.Chem.rdchem.Mol with 3D coordinates. This is a computationally intensive operation that may take a long time for complicated molecules. |
|  `mmff_optimize_molecule_max_iterations` |  Number of iterations to use in rdkit.Chem.AllChem.MMFFOptimizeMolecule. |

<!-- lazydoc-ignore-classmethod: internal -->


| Class Variables |  |
| :--- | :--- |
|  `SUPPORTED_RDKIT_TYPES`<a id="SUPPORTED_RDKIT_TYPES"></a> |   |
|  `SUPPORTED_TYPES`<a id="SUPPORTED_TYPES"></a> |   |
