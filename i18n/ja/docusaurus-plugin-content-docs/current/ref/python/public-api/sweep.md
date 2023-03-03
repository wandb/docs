# Sweep



[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)View source on GitHub](https://www.github.com/wandb/client/tree/1725d84a5bc68d5ecf9aedcbcc447e7e2fb1a1cf/wandb/apis/public.py#L2483-L2693)



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



[View source](https://www.github.com/wandb/client/tree/1725d84a5bc68d5ecf9aedcbcc447e7e2fb1a1cf/wandb/apis/public.py#L2578-L2601)

```python
best_run(
 order=None
)
```

Returns the best run sorted by the metric defined in config or the order passed in


### `display`



[View source](https://www.github.com/wandb/client/tree/1725d84a5bc68d5ecf9aedcbcc447e7e2fb1a1cf/wandb/apis/public.py#L971-L982)

```python
display(
 height=420, hidden=(False)
) -> bool
```

Display this object in jupyter


### `get`



[View source](https://www.github.com/wandb/client/tree/1725d84a5bc68d5ecf9aedcbcc447e7e2fb1a1cf/wandb/apis/public.py#L2626-L2675)

```python
@classmethod
get(
 client, entity=None, project=None, sid=None, order=None, query=None, **kwargs
)
```

Execute a query against the cloud backend


### `load`



[View source](https://www.github.com/wandb/client/tree/1725d84a5bc68d5ecf9aedcbcc447e7e2fb1a1cf/wandb/apis/public.py#L2559-L2567)

```python
load(
 force: bool = (False)
)
```




### `snake_to_camel`



[View source](https://www.github.com/wandb/client/tree/1725d84a5bc68d5ecf9aedcbcc447e7e2fb1a1cf/wandb/apis/public.py#L967-L969)

```python
snake_to_camel(
 string
)
```




### `to_html`



[View source](https://www.github.com/wandb/client/tree/1725d84a5bc68d5ecf9aedcbcc447e7e2fb1a1cf/wandb/apis/public.py#L2677-L2685)

```python
to_html(
 height=420, hidden=(False)
)
```

Generate HTML containing an iframe displaying this sweep






| Class Variables | |
| :--- | :--- |
| `LEGACY_QUERY` | |
| `QUERY` | |

