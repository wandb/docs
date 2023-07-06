# Project



[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)View source on GitHub](https://www.github.com/wandb/client/tree/v0.15.5/wandb/apis/public.py#L1440-L1522)



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



[View source](https://www.github.com/wandb/client/tree/v0.15.5/wandb/apis/public.py#L1473-L1475)

```python
artifacts_types(
 per_page=50
)
```




### `display`



[View source](https://www.github.com/wandb/client/tree/v0.15.5/wandb/apis/public.py#L954-L965)

```python
display(
 height=420, hidden=(False)
) -> bool
```

Display this object in jupyter.


### `snake_to_camel`



[View source](https://www.github.com/wandb/client/tree/v0.15.5/wandb/apis/public.py#L950-L952)

```python
snake_to_camel(
 string
)
```




### `sweeps`



[View source](https://www.github.com/wandb/client/tree/v0.15.5/wandb/apis/public.py#L1477-L1522)

```python
sweeps()
```




### `to_html`



[View source](https://www.github.com/wandb/client/tree/v0.15.5/wandb/apis/public.py#L1457-L1465)

```python
to_html(
 height=420, hidden=(False)
)
```

Generate HTML containing an iframe displaying this project.




