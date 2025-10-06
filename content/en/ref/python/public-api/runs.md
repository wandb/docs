---
title: Runs
namespace: public_apis_namespace
python_object_type: class
---
{{< readfile file="/_includes/public-api-use.md" >}}


{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/apis/public/runs.py >}}




## <kbd>class</kbd> `Runs`
A lazy iterator of `Run` objects associated with a project and optional filter. 

Runs are retrieved in pages from the W&B server as needed. 

This is generally used indirectly using the `Api.runs` namespace.

### <kbd>method</kbd> `Runs.__init__`

```python
__init__(
    client: 'RetryingClient',
    entity: 'str',
    project: 'str',
    filters: 'dict[str, Any] | None' = None,
    order: 'str' = '+created_at',
    per_page: 'int' = 50,
    include_sweeps: 'bool' = True
)
```

**Args:**
 
 - `client`:  (`wandb.apis.public.RetryingClient`) The API client to use  for requests. 
 - `entity`:  (str) The entity (username or team) that owns the project. 
 - `project`:  (str) The name of the project to fetch runs from. 
 - `filters`:  (Optional[Dict[str, Any]]) A dictionary of filters to apply  to the runs query. 
 - `order`:  (str) Order can be `created_at`, `heartbeat_at`, `config.*.value`, or `summary_metrics.*`.  If you prepend order with a + order is ascending (default).  If you prepend order with a - order is descending.  The default order is run.created_at from oldest to newest. 
 - `per_page`:  (int) The number of runs to fetch per request (default is 50). 
 - `include_sweeps`:  (bool) Whether to include sweep information in the  runs. Defaults to True. 



**Examples:**
 ```python
from wandb.apis.public.runs import Runs
from wandb.apis.public import Api

# Get all runs from a project that satisfy the filters
filters = {"state": "finished", "config.optimizer": "adam"}

runs = Api().runs(
    client=api.client,
    entity="entity",
    project="project_name",
    filters=filters,
)

# Iterate over runs and print details
for run in runs:
    print(f"Run name: {run.name}")
    print(f"Run ID: {run.id}")
    print(f"Run URL: {run.url}")
    print(f"Run state: {run.state}")
    print(f"Run config: {run.config}")
    print(f"Run summary: {run.summary}")
    print(f"Run history (samples=5): {run.history(samples=5)}")
    print("----------")

# Get histories for all runs with specific metrics
histories_df = runs.histories(
    samples=100,  # Number of samples per run
    keys=["loss", "accuracy"],  # Metrics to fetch
    x_axis="_step",  # X-axis metric
    format="pandas",  # Return as pandas DataFrame
)
``` 







---


### <kbd>property</kbd> Runs.length





---



### <kbd>method</kbd> `Runs.histories`

```python
histories(
    samples: 'int' = 500,
    keys: 'list[str] | None' = None,
    x_axis: 'str' = '_step',
    format: "Literal['default', 'pandas', 'polars']" = 'default',
    stream: "Literal['default', 'system']" = 'default'
)
```

Return sampled history metrics for all runs that fit the filters conditions. 



**Args:**
 
 - `samples`:  The number of samples to return per run 
 - `keys`:  Only return metrics for specific keys 
 - `x_axis`:  Use this metric as the xAxis defaults to _step 
 - `format`:  Format to return data in, options are "default", "pandas",  "polars" 
 - `stream`:  "default" for metrics, "system" for machine metrics 

**Returns:**
 
 - `pandas.DataFrame`:  If `format="pandas"`, returns a `pandas.DataFrame`  of history metrics. 
 - `polars.DataFrame`:  If `format="polars"`, returns a `polars.DataFrame`  of history metrics. 
 - `list of dicts`:  If `format="default"`, returns a list of dicts  containing history metrics with a `run_id` key. 

