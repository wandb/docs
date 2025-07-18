---
title: sweep()
object_type: python_sdk_actions
data_type_classification: function
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_sweep.py >}}




### <kbd>function</kbd> `sweep`

```python
sweep(
    sweep: Union[dict, Callable],
    entity: Optional[str] = None,
    project: Optional[str] = None,
    prior_runs: Optional[List[str]] = None
) → str
```

Initialize a hyperparameter sweep. 

Search for hyperparameters that optimizes a cost function of a machine learning model by testing various combinations. 

Make note the unique identifier, `sweep_id`, that is returned. At a later step provide the `sweep_id` to a sweep agent. 

See [Sweep configuration structure](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration) for information on how to define your sweep. 



**Args:**
 
 - `sweep`:  The configuration of a hyperparameter search.  (or configuration generator).  If you provide a callable, ensure that the callable does  not take arguments and that it returns a dictionary that  conforms to the W&B sweep config spec. 
 - `entity`:  The username or team name where you want to send W&B  runs created by the sweep to. Ensure that the entity you  specify already exists. If you don't specify an entity,  the run will be sent to your default entity,  which is usually your username. 
 - `project`:  The name of the project where W&B runs created from  the sweep are sent to. If the project is not specified, the  run is sent to a project labeled 'Uncategorized'. 
 - `prior_runs`:  The run IDs of existing runs to add to this sweep. 



**Returns:**
 
 - `sweep_id`:  (str) A unique identifier for the sweep. 
