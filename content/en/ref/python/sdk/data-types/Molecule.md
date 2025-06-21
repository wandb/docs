---
title: Molecule
object_type: python_sdk_data_type
data_type_classification: class
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/wandb/sdk/data_types/molecule.py >}}




## <kbd>class</kbd> `Molecule`
W&B class for 3D Molecular data. 



**Args:**
 
 - `data_or_path`:  (pathlib.Path, string, io)  Molecule can be initialized from a file name or an io object. 
 - `caption`:  (string)  Caption associated with the molecule for display. 

### <kbd>method</kbd> `Molecule.__init__`

```python
__init__(
    data_or_path: Union[str, pathlib.Path, ForwardRef('TextIO')],
    caption: Optional[str] = None,
    **kwargs: str
) → None
```








---

### <kbd>classmethod</kbd> `Molecule.from_rdkit`

```python
from_rdkit(
    data_or_path: 'RDKitDataType',
    caption: Optional[str] = None,
    convert_to_3d_and_optimize: bool = True,
    mmff_optimize_molecule_max_iterations: int = 200
) → Molecule
```

Convert RDKit-supported file/object types to wandb.Molecule. 



**Args:**
 
 - `data_or_path`:  (string, rdkit.Chem.rdchem.Mol)  Molecule can be initialized from a file name or an rdkit.Chem.rdchem.Mol object. 
 - `caption`:  (string)  Caption associated with the molecule for display. 
 - `convert_to_3d_and_optimize`:  (bool)  Convert to rdkit.Chem.rdchem.Mol with 3D coordinates.  This is an expensive operation that may take a long time for complicated molecules. 
 - `mmff_optimize_molecule_max_iterations`:  (int)  Number of iterations to use in rdkit.Chem.AllChem.MMFFOptimizeMolecule 

<!-- lazydoc-ignore: internal --> 

---

### <kbd>classmethod</kbd> `Molecule.from_smiles`

```python
from_smiles(
    data: str,
    caption: Optional[str] = None,
    sanitize: bool = True,
    convert_to_3d_and_optimize: bool = True,
    mmff_optimize_molecule_max_iterations: int = 200
) → Molecule
```

Convert SMILES string to wandb.Molecule. 



**Args:**
 
 - `data`:  SMILES string. 
 - `caption`:  Caption associated with the molecule for display. 
 - `sanitize`:  Check if the molecule is chemically reasonable by  the RDKit's definition. 
 - `convert_to_3d_and_optimize`:  Convert to rdkit.Chem.rdchem.Mol  with 3D coordinates. This is a computationally intensive  operation that may take a long time for complicated molecules. 
 - `mmff_optimize_molecule_max_iterations`:  Number of iterations to  use in rdkit.Chem.AllChem.MMFFOptimizeMolecule. 

<!-- lazydoc-ignore: internal --> 

---

### <kbd>classmethod</kbd> `Molecule.get_media_subdir`

```python
get_media_subdir() → str
```

Get media subdirectory. 

<!-- lazydoc-ignore: internal --> 

---

### <kbd>classmethod</kbd> `Molecule.seq_to_json`

```python
seq_to_json(
    seq: Sequence[ForwardRef('BatchableMedia')],
    run: 'LocalRun',
    key: str,
    step: Union[int, str]
) → dict
```

Convert a sequence of Molecule objects to a JSON representation. 

<!-- lazydoc-ignore: internal --> 

---

