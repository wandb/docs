---
title: Sweep
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/public/sweeps.py#L197-L441 >}}

The set of runs associated with the sweep.

| Attributes |  |
| :--- | :--- |
|  `config` |  The sweep configuration used for the sweep. |
|  `entity` |  The entity associated with the sweep. |
|  `expected_run_count` |  Return the number of expected runs in the sweep or None for infinite runs. |
|  `name` |  The name of the sweep. Returns the first name that exists in the following priority order: 1. User-edited display name 2. Name configured at creation time 3. Sweep ID |
|  `order` |  Return the order key for the sweep. |
|  `path` |  Returns the path of the project. The path is a list containing the entity, project name, and sweep ID. |
|  `url` |  The URL of the sweep. The sweep URL is generated from the entity, project, the term "sweeps", and the sweep ID.run_id. For SaaS users, it takes the form of `https://wandb.ai/entity/project/sweeps/sweeps_ID`. |
|  `username` |  Deprecated. Use `Sweep.entity` instead. |

## Methods

### `best_run`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/public/sweeps.py#L295-L318)

```python
best_run(
    order=None
)
```

Return the best run sorted by the metric defined in config or the order passed in.

### `display`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/attrs.py#L18-L38)

```python
display(
    height=420, hidden=(False)
) -> bool
```

Display this object in jupyter.

### `get`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/public/sweeps.py#L361-L423)

```python
@classmethod
get(
    client: RetryingClient,
    entity: (str | None) = None,
    project: (str | None) = None,
    sid: (str | None) = None,
    order: (str | None) = None,
    query: (str | None) = None,
    **kwargs
)
```

Execute a query against the cloud backend.

| Args |  |
| :--- | :--- |
|  `client` |  The client to use to execute the query. |
|  `entity` |  The entity (username or team) that owns the project. |
|  `project` |  The name of the project to fetch sweep from. |
|  `sid` |  The sweep ID to query. |
|  `order` |  The order in which the sweep's runs are returned. |
|  `query` |  The query to use to execute the query. |
|  `**kwargs` |  Additional keyword arguments to pass to the query. |

### `load`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/public/sweeps.py#L271-L283)

```python
load(
    force: bool = (False)
)
```

Fetch and update sweep data logged to the run from GraphQL database.

<!-- lazydoc-ignore: internal -->


### `snake_to_camel`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/attrs.py#L14-L16)

```python
snake_to_camel(
    string
)
```

### `to_html`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/public/sweeps.py#L425-L433)

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
