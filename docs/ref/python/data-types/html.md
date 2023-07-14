# Html



[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)View source on GitHub](https://www.github.com/wandb/client/tree/v0.15.5/wandb/sdk/data_types/html.py#L18-L107)



Wandb class for arbitrary html.

```python
Html(
 data: Union[str, 'TextIO'],
 inject: bool = (True)
) -> None
```





| Arguments | |
| :--- | :--- |
| `data` | (string or io object) HTML to display in wandb |
| `inject` | (boolean) Add a stylesheet to the HTML object. If set to False the HTML will pass through unchanged. |



## Methods

### `inject_head`



[View source](https://www.github.com/wandb/client/tree/v0.15.5/wandb/sdk/data_types/html.py#L59-L74)

```python
inject_head() -> None
```






