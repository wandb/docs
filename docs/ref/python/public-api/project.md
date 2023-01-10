# Project



[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)View source on GitHub](https://www.github.com/wandb/client/tree/597de7d094bdab2fa17d5db396c6bc227b2f62c3/wandb/apis/public.py#L1446-L1528)



A project is a namespace for runs.

```python
Project(
 client, entity, project, attrs
)
```







| Attributes | |
| :--- | :--- |



## Methods

### `artifacts_types`



[View source](https://www.github.com/wandb/client/tree/597de7d094bdab2fa17d5db396c6bc227b2f62c3/wandb/apis/public.py#L1479-L1481)

```python
artifacts_types(
 per_page=50
)
```




### `display`



[View source](https://www.github.com/wandb/client/tree/597de7d094bdab2fa17d5db396c6bc227b2f62c3/wandb/apis/public.py#L959-L970)

```python
display(
 height=420, hidden=(False)
) -> bool
```

Display this object in jupyter


### `snake_to_camel`



[View source](https://www.github.com/wandb/client/tree/597de7d094bdab2fa17d5db396c6bc227b2f62c3/wandb/apis/public.py#L955-L957)

```python
snake_to_camel(
 string
)
```




### `sweeps`



[View source](https://www.github.com/wandb/client/tree/597de7d094bdab2fa17d5db396c6bc227b2f62c3/wandb/apis/public.py#L1483-L1528)

```python
sweeps()
```




### `to_html`



[View source](https://www.github.com/wandb/client/tree/597de7d094bdab2fa17d5db396c6bc227b2f62c3/wandb/apis/public.py#L1463-L1471)

```python
to_html(
 height=420, hidden=(False)
)
```

Generate HTML containing an iframe displaying this project




