---
title: Sweep
---

{{< cta-button githubLink="https://www.github.com/wandb/wandb/tree/v0.18.7/wandb/apis/public/sweeps.py#L30-L240" >}}

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
sweep = api.sweep(path / to / sweep)
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

[View source](https://www.github.com/wandb/wandb/tree/v0.19.1/wandb/apis/public/sweeps.py#L125-L148)

```python
best_run(
    order=None
)
```

Return the best run sorted by the metric defined in config or the order passed in.

### `display`

[View source](https://www.github.com/wandb/wandb/tree/v0.19.1/wandb/apis/attrs.py#L16-L37)

```python
display(
    height=420, hidden=(False)
) -> bool
```

Display this object in jupyter.

### `get`

[View source](https://www.github.com/wandb/wandb/tree/v0.19.1/wandb/apis/public/sweeps.py#L173-L222)

```python
@classmethod
get(
    client, entity=None, project=None, sid=None, order=None, query=None, **kwargs
)
```

Execute a query against the cloud backend.

### `load`

[View source](https://www.github.com/wandb/wandb/tree/v0.19.1/wandb/apis/public/sweeps.py#L106-L114)

```python
load(
    force: bool = (False)
)
```

### `snake_to_camel`

[View source](https://www.github.com/wandb/wandb/tree/v0.19.1/wandb/apis/attrs.py#L12-L14)

```python
snake_to_camel(
    string
)
```

### `to_html`

[View source](https://www.github.com/wandb/wandb/tree/v0.19.1/wandb/apis/public/sweeps.py#L224-L232)

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
