# controller



[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)View source on GitHub](https://www.github.com/wandb/client/tree/9f1a662d681e96387ebf650900aef8f19703b575/wandb/sdk/wandb_sweep.py#L119-L143)



Public sweep controller constructor.

```python
controller(
 sweep_id_or_config: Optional[Union[str, Dict]] = None,
 entity: Optional[str] = None,
 project: Optional[str] = None
)
```





#### Usage:

```python
import wandb

tuner = wandb.controller(...)
print(tuner.sweep_config)
print(tuner.sweep_id)
tuner.configure_search(...)
tuner.configure_stopping(...)
```
