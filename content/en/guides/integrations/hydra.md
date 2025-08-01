---
description: How to integrate W&B with Hydra.
menu:
  default:
    identifier: hydra
    parent: integrations
title: Hydra
weight: 150
---

> [Hydra](https://hydra.cc) is an open-source Python framework that simplifies the development of research and other complex applications. The key feature is the ability to dynamically create a hierarchical configuration by composition and override it through config files and the command line.

You can continue to use Hydra for configuration management while taking advantage of the power of W&B.

## Track metrics

Track your metrics as normal with `wandb.init()` and `wandb.Run.log()` . Here, `wandb.entity` and `wandb.project` are defined within a hydra configuration file.

```python
import wandb


@hydra.main(config_path="configs/", config_name="defaults")
def run_experiment(cfg):

    with wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project) as run:
      run.log({"loss": loss})
```

## Track Hyperparameters

Hydra uses [omegaconf](https://omegaconf.readthedocs.io/en/2.1_branch/) as the default way to interface with configuration dictionaries. `OmegaConf`'s dictionary are not a subclass of primitive dictionaries so directly passing Hydra's `Config` to `wandb.Run.config` leads to unexpected results on the dashboard. It's necessary to convert `omegaconf.DictConfig` to the primitive `dict` type before passing to `wandb.Run.config`.

```python
@hydra.main(config_path="configs/", config_name="defaults")
def run_experiment(cfg):
  with wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project) as run:
    run.config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project)
    run.log({"loss": loss})
    model = Model(**run.config.model.configs)
```

## Troubleshoot multiprocessing

If your process hangs when started, this may be caused by [this known issue]({{< relref "/guides/models/track/log/distributed-training.md" >}}). To solve this, try to changing wandb's multiprocessing protocol either by adding an extra settings parameter to `wandb.init()` as:

```python
wandb.init(settings=wandb.Settings(start_method="thread"))
```

or by setting a global environment variable from your shell:

```bash
$ export WANDB_START_METHOD=thread
```

## Optimize Hyperparameters

[W&B Sweeps]({{< relref "/guides/models/sweeps/" >}}) is a highly scalable hyperparameter search platform, which provides interesting insights and visualization about W&B experiments with minimal requirements code real-estate. Sweeps integrates seamlessly with Hydra projects with no-coding requirements. The only thing needed is a configuration file describing the various parameters to sweep over as normal.

A simple example `sweep.yaml` file would be:

```yaml
program: main.py
method: bayes
metric:
  goal: maximize
  name: test/accuracy
parameters:
  dataset:
    values: [mnist, cifar10]

command:
  - ${env}
  - python
  - ${program}
  - ${args_no_hyphens}
```

Invoke the sweep:

``` bash
wandb sweep sweep.yaml` \
```

W&B automatically creates a sweep inside your project and returns a `wandb agent` command for you to run on each machine you want to run your sweep.

### Pass parameters not present in Hydra defaults

<a id="pitfall-3-sweep-passing-parameters-not-present-in-defaults"></a>

Hydra supports passing extra parameters through the command line which aren't present in the default configuration file, by using a `+` before command. For example, you can pass an extra parameter with some value by simply calling:

```bash
$ python program.py +experiment=some_experiment
```

You cannot sweep over such `+` configurations similar to what one does while configuring [Hydra Experiments](https://hydra.cc/docs/patterns/configuring_experiments/). To work around this, you can initialize the experiment parameter with a default empty file and use W&B Sweep to override those empty configs on each call. For more information, read [this W&B Report](https://wandb.me/hydra).
