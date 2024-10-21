---
title: "Do I need to provide values for all hyperparameters as part of the W&B Sweep. Can I set defaults?"
tags:
   - sweeps
---

The hyperparameter names and values specified as part of the sweep configuration are accessible in `wandb.config`, a dictionary-like object.

For runs that are not part of a sweep, the values of `wandb.config` are usually set by providing a dictionary to the `config` argument of `wandb.init`. During a sweep, however, any configuration information passed to `wandb.init` is instead treated as a default value, which might be over-ridden by the sweep.

You can also be more explicit about the intended behavior by using `config.setdefaults`. Code snippets for both methods appear below:

<Tabs
  defaultValue="wandb.init"
  values={[
    {label: 'wandb.init', value: 'wandb.init'},
    {label: 'config.setdefaults', value: 'config.setdef'},
  ]}>
  <TabItem value="wandb.init">

```python
# set default values for hyperparameters
config_defaults = {"lr": 0.1, "batch_size": 256}

# start a run, providing defaults
#   that can be over-ridden by the sweep
with wandb.init(config=config_default) as run:
    # add your training code here
    ...
```

  </TabItem>
  <TabItem value="config.setdef">

```python
# set default values for hyperparameters
config_defaults = {"lr": 0.1, "batch_size": 256}

# start a run
with wandb.init() as run:
    # update any values not set by sweep
    run.config.setdefaults(config_defaults)

    # add your training code here
```

  </TabItem>
</Tabs>