# Sweep



[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)View source on GitHub](https://www.github.com/wandb/client/tree/9f1a662d681e96387ebf650900aef8f19703b575/wandb/apis/public.py#L2487-L2697)



A set of runs associated with a sweep.

```python
Sweep(
 client, entity, project, sweep_id, attrs=None
)
```





#### Examples:

Instantiate with:
```
api = wandb.Api()
sweep = api.sweep(path/to/sweep)
```





| Attributes | |
| :--- | :--- |
| `runs` | (`Runs`) list of runs |
| `id` | (str) sweep id |
| `project` | (str) name of project |
| `config` | (str) dictionary of sweep configuration |
| `state` | (str) the state of the sweep |
| `expected_run_count` | (int) number of expected runs for the sweep |



## Methods

### `best_run`



[View source](https://www.github.com/wandb/client/tree/9f1a662d681e96387ebf650900aef8f19703b575/wandb/apis/public.py#L2582-L2605)

```python
best_run(
 order=None
)
```

Return the best run sorted by the metric defined in config or the order passed in.


### `display`



[View source](https://www.github.com/wandb/client/tree/9f1a662d681e96387ebf650900aef8f19703b575/wandb/apis/public.py#L979-L990)

```python
display(
 height=420, hidden=(False)
) -> bool
```

Display this object in jupyter.


### `get`



[View source](https://www.github.com/wandb/client/tree/9f1a662d681e96387ebf650900aef8f19703b575/wandb/apis/public.py#L2630-L2679)

```python
@classmethod
get(
 client, entity=None, project=None, sid=None, order=None, query=None, **kwargs
)
```

Execute a query against the cloud backend.


### `load`



[View source](https://www.github.com/wandb/client/tree/9f1a662d681e96387ebf650900aef8f19703b575/wandb/apis/public.py#L2563-L2571)

```python
load(
 force: bool = (False)
)
```




### `snake_to_camel`



[View source](https://www.github.com/wandb/client/tree/9f1a662d681e96387ebf650900aef8f19703b575/wandb/apis/public.py#L975-L977)

```python
snake_to_camel(
 string
)
```




### `to_html`



[View source](https://www.github.com/wandb/client/tree/9f1a662d681e96387ebf650900aef8f19703b575/wandb/apis/public.py#L2681-L2689)

```python
to_html(
 height=420, hidden=(False)
)
```

Generate HTML containing an iframe displaying this sweep.






| Class Variables | |
| :--- | :--- |
| `LEGACY_QUERY` | |
| `QUERY` | |

