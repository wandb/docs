# Project



[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)View source on GitHub](https://www.github.com/wandb/client/tree/c505c66a5f9c1530671564dae3e9e230f72f6584/wandb/apis/public.py#L1465-L1547)



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



[View source](https://www.github.com/wandb/client/tree/c505c66a5f9c1530671564dae3e9e230f72f6584/wandb/apis/public.py#L1498-L1500)

```python
artifacts_types(
 per_page=50
)
```




### `display`



[View source](https://www.github.com/wandb/client/tree/c505c66a5f9c1530671564dae3e9e230f72f6584/wandb/apis/public.py#L979-L990)

```python
display(
 height=420, hidden=(False)
) -> bool
```

Display this object in jupyter.


### `snake_to_camel`



[View source](https://www.github.com/wandb/client/tree/c505c66a5f9c1530671564dae3e9e230f72f6584/wandb/apis/public.py#L975-L977)

```python
snake_to_camel(
 string
)
```




### `sweeps`



[View source](https://www.github.com/wandb/client/tree/c505c66a5f9c1530671564dae3e9e230f72f6584/wandb/apis/public.py#L1502-L1547)

```python
sweeps()
```




### `to_html`



[View source](https://www.github.com/wandb/client/tree/c505c66a5f9c1530671564dae3e9e230f72f6584/wandb/apis/public.py#L1482-L1490)

```python
to_html(
 height=420, hidden=(False)
)
```

Generate HTML containing an iframe displaying this project.




