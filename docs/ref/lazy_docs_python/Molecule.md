import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx'

# Molecule

<CTAButtons githubLink='https://github.com/wandb/wandb/blob/main/wandb/sdk/data_types/molecule.py'/>




## <kbd>class</kbd> `Molecule`
Wandb class for 3D Molecular data. 



**Arguments:**
 
 - `data_or_path`:  (string, io)  Molecule can be initialized from a file name or an io object. 
 - `caption`:  (string)  Caption associated with the molecule for display. 

### <kbd>method</kbd> `Molecule.__init__`

```python
__init__(
    data_or_path: Union[str, ForwardRef('TextIO')],
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



**Arguments:**
 
 - `data_or_path`:  (string, rdkit.Chem.rdchem.Mol)  Molecule can be initialized from a file name or an rdkit.Chem.rdchem.Mol object. 
 - `caption`:  (string)  Caption associated with the molecule for display. 
 - `convert_to_3d_and_optimize`:  (bool)  Convert to rdkit.Chem.rdchem.Mol with 3D coordinates.  This is an expensive operation that may take a long time for complicated molecules. 
 - `mmff_optimize_molecule_max_iterations`:  (int)  Number of iterations to use in rdkit.Chem.AllChem.MMFFOptimizeMolecule 

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



**Arguments:**
 
 - `data`:  (string)  SMILES string. 
 - `caption`:  (string)  Caption associated with the molecule for display 
 - `sanitize`:  (bool)  Check if the molecule is chemically reasonable by the RDKit's definition. 
 - `convert_to_3d_and_optimize`:  (bool)  Convert to rdkit.Chem.rdchem.Mol with 3D coordinates.  This is an expensive operation that may take a long time for complicated molecules. 
 - `mmff_optimize_molecule_max_iterations`:  (int)  Number of iterations to use in rdkit.Chem.AllChem.MMFFOptimizeMolecule 

---

### <kbd>classmethod</kbd> `Molecule.get_media_subdir`

```python
get_media_subdir() → str
```





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





---

### <kbd>method</kbd> `Molecule.to_json`

```python
to_json(
    run_or_artifact: Union[ForwardRef('LocalRun'), ForwardRef('Artifact')]
) → dict
```