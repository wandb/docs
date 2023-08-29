# Sweep

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.15.9/wandb/apis/public.py#L2806-L3016' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>


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

| Attributes |  |
| :--- | :--- |
|  `runs` |  (`Runs`) list of runs |
|  `id` |  (str) sweep id |
|  `project` |  (str) name of project |
|  `config` |  (str) dictionary of sweep configuration |
|  `state` |  (str) the state of the sweep |
|  `expected_run_count` |  (int) number of expected runs for the sweep |

## Methods

### `best_run`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.9/wandb/apis/public.py#L2901-L2924)

```python
best_run(
    order=None
)
```

Return the best run sorted by the metric defined in config or the order passed in.

### `display`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.9/wandb/apis/public.py#L1131-L1142)

```python
display(
    height=420, hidden=(False)
) -> bool
```

Display this object in jupyter.

### `get`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.9/wandb/apis/public.py#L2949-L2998)

```python
@classmethod
get(
    client, entity=None, project=None, sid=None, order=None, query=None, **kwargs
)
```

Execute a query against the cloud backend.

### `load`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.9/wandb/apis/public.py#L2882-L2890)

```python
load(
    force: bool = (False)
)
```

### `snake_to_camel`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.9/wandb/apis/public.py#L1127-L1129)

```python
snake_to_camel(
    string
)
```

### `to_html`

[View source](https://www.github.com/wandb/wandb/tree/v0.15.9/wandb/apis/public.py#L3000-L3008)

```python
to_html(
    height=420, hidden=(False)
)
```

Generate HTML containing an iframe displaying this sweep.

| Class Variables |  |
| :--- | :--- |
|  `LEGACY_QUERY`<a id="LEGACY_QUERY"></a> |   |
|  `QUERY`<a id="QUERY"></a> |   |
