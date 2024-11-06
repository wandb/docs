# watch

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.18.6/wandb/sdk/wandb_run.py#L2869-L2904' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>


Hooks into the given PyTorch model(s) to monitor gradients and the model's computational graph.

```python
watch(
    models: (torch.nn.Module | Sequence[torch.nn.Module]),
    criterion: (torch.F | None) = None,
    log: (Literal['gradients', 'parameters', 'all'] | None) = "gradients",
    log_freq: int = 1000,
    idx: (int | None) = None,
    log_graph: bool = (False)
) -> None
```

This function can track parameters, gradients, or both during training. It should be
extended to support arbitrary machine learning models in the future.

| Args |  |
| :--- | :--- |
|  models (Union[torch.nn.Module, Sequence[torch.nn.Module]]): A single model or a sequence of models to be monitored. criterion (Optional[torch.F]): The loss function being optimized (optional). log (Optional[Literal["gradients", "parameters", "all"]]): Specifies whether to log "gradients", "parameters", or "all". Set to None to disable logging. (default="gradients") log_freq (int): Frequency (in batches) to log gradients and parameters. (default=1000) idx (Optional[int]): Index used when tracking multiple models with `wandb.watch`. (default=None) log_graph (bool): Whether to log the model's computational graph. (default=False) |

| Raises |  |
| :--- | :--- |
|  `ValueError` |  If `wandb.init` has not been called or if any of the models are not instances of `torch.nn.Module`. |
