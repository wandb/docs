import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx'

# define_metric

<CTAButtons githubLink='https://github.com/wandb/wandb/blob/main/wandb/sdk/lib/preinit.py'/>




### <kbd>function</kbd> `wandb.define_metric`

```python
wandb.define_metric(
    name: 'str',
    step_metric: 'str | wandb_metric.Metric | None' = None,
    step_sync: 'bool | None' = None,
    hidden: 'bool | None' = None,
    summary: 'str | None' = None,
    goal: 'str | None' = None,
    overwrite: 'bool | None' = None
) → wandb_metric.Metric
```

Customize metrics logged with `wandb.log()`. 



**Args:**
 
 - `name`:  The name of the metric to customize. 
 - `step_metric`:  The name of another metric to serve as the X-axis  for this metric in automatically generated charts. 
 - `step_sync`:  Automatically insert the last value of step_metric into  `run.log()` if it is not provided explicitly. Defaults to True  if step_metric is specified. 
 - `hidden`:  Hide this metric from automatic plots. 
 - `summary`:  Specify aggregate metrics added to summary.  Supported aggregations include "min", "max", "mean", "last",  "best", "copy" and "none". "best" is used together with the  goal parameter. "none" prevents a summary from being generated.  "copy" is deprecated and should not be used. 
 - `goal`:  Specify how to interpret the "best" summary type.  Supported options are "minimize" and "maximize". 
 - `overwrite`:  If false, then this call is merged with previous  `define_metric` calls for the same metric by using their  values for any unspecified parameters. If true, then  unspecified parameters overwrite values specified by  previous calls. 



**Returns:**
 An object that represents this call but can otherwise be discarded.