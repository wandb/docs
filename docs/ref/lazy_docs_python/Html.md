import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx'

# Html

<CTAButtons githubLink='https://github.com/wandb/wandb/blob/main/wandb/sdk/data_types/html.py'/>




## <kbd>class</kbd> `Html`
Wandb class for arbitrary html. 



**Arguments:**
 
 - `data`:  (string or io object) HTML to display in wandb 
 - `inject`:  (boolean) Add a stylesheet to the HTML object.  If set  to False the HTML will pass through unchanged. 

### <kbd>method</kbd> `Html.__init__`

```python
__init__(data: Union[str, ForwardRef('TextIO')], inject: bool = True) → None
```








---

### <kbd>classmethod</kbd> `Html.from_json`

```python
from_json(json_obj: dict, source_artifact: 'Artifact') → Html
```





---

### <kbd>classmethod</kbd> `Html.get_media_subdir`

```python
get_media_subdir() → str
```





---

### <kbd>method</kbd> `Html.inject_head`

```python
inject_head() → None
```





---

### <kbd>classmethod</kbd> `Html.seq_to_json`

```python
seq_to_json(
    seq: Sequence[ForwardRef('BatchableMedia')],
    run: 'LocalRun',
    key: str,
    step: Union[int, str]
) → dict
```





---

### <kbd>method</kbd> `Html.to_json`

```python
to_json(
    run_or_artifact: Union[ForwardRef('LocalRun'), ForwardRef('Artifact')]
) → dict
```