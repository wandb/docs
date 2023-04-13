
# 🌞 Ray/Tune and 🏋️‍♀️ Weights & Biases 

<a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/raytune/RayTune_with_wandb.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

Both Weights and Biases and Ray/Tune are built for scale and handle millions of models every month for teams doing some of the most cutting-edge deep learning research.

[W&B](https://wandb.com) is a toolkit with everything you need to track, reproduce, and gain insights from your models easily; [Ray/Tune](https://docs.ray.io/en/latest/tune/) provides a simple interface for scaling and running distributed experiments.

### 🤝 They're a natural match! 🤝

Here's just a few reasons why our community likes Ray/Tune –

* **Simple distributed execution**: Ray/Tune makes it easy to scale all the way from a single node on a laptop, through to multiple GPUs, and up to multiple nodes on multiple machines.
* **State-of-the-art algorithms**: Ray/Tune has tested implementations of a huge number of potent scheduling algorithms including
[Population-Based Training](https://docs.ray.io/en/latest/tune/tutorials/tune-advanced-tutorial.html),
[ASHA](https://docs.ray.io/en/master/tune/tutorials/tune-tutorial.html#early-stopping-with-asha),
and
[HyperBand](https://docs.ray.io/en/latest/tune/api_docs/schedulers.html#hyperband-tune-schedulers-hyperbandscheduler)
* **Method agnostic**: Ray/Tune works across deep learning frameworks (including PyTorch, Keras, Tensorflow, and PyTorchLightning) and with other ML methods like gradient-boosted trees (XGBoost, LightGBM)
* **Fault-tolerance**: Ray/Tune is built on top of Ray, providing tolerance for failed runs out of the box.

This Colab demonstrates how this integration works for a simple grid search over two hyperparameters. If you've got any questions about the details,
check out
[our documentation](https://docs.wandb.com/library/integrations/ray-tune)
or the
[documentation for Ray/Tune](https://docs.ray.io/en/master/tune/api_docs/integration.html#weights-and-biases-tune-integration-wandb).


W&B integrates with `ray.tune` by offering two lightweight standalone integrations:

1. For simple cases, `WandbLoggerCallback` automatically logs metrics reported to Tune to W&B, along with the configuration of the experiment, using Tune's [`logger` interface](https://docs.ray.io/en/latest/tune/api_docs/logging.html).
2. The `@wandb_mixin` decorator gives you greater control over logging by letting you call `wandb.log` inside the decorated function, allowing you to [log custom metrics, plots, and other outputs, like media](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Log_(Almost)_Anything_with_W%26B_Media.ipynb).

These methods can be used together or independently.

The example below demonstrates how they can be used together.

# 🧹 Running a hyperparameter sweep with W&B and Ray/Tune

## 📥 Install, `import`, and set seeds


Let's start by installing the libraries and importing everything we need.



```
!pip install -Uq ray[tune] wandb
```


```
import random
import numpy as np
from ray import tune
from ray.tune.logger import DEFAULT_LOGGERS
from ray.air.callbacks.wandb import WandbLoggerCallback
import torch
import torch.optim as optim
import wandb
```


```
wandb.login()
```

We'll make use of Ray's handy [`mnist_pytorch` example code](https://github.com/ray-project/ray/blob/master/python/ray/tune/examples/mnist_pytorch.py).


```
from ray.tune.examples.mnist_pytorch import ConvNet, get_data_loaders, test, train
```

In order to make this experiment reproducible, we'll set the seeds for random number generators of various libraries used in this experiment.


```
torch.backends.cudnn.deterministic = True
random.seed(2022)
np.random.seed(2022)
torch.manual_seed(2022)
torch.cuda.manual_seed_all(2022)
```

## 🤝 Integrating W&B with Ray/Tune

Now, we define our training process, decorated with `@wandb_mixin` so we can call `wandb.log` to log our custom metric
(here, just the error rate; you might also [log media](https://docs.wandb.com/library/log#media), e.g. images from the validation set, captioned by the model predictions).

When we execute our hyperparameter sweep below,
this function will be called with a `config`uration dictionary
that contains values for any hyperparameters.
For simplicity, we only have two hyperparameters here:
the learning rate and momentum value for accelerated SGD.


```
@wandb_mixin
def train_mnist(config):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_data_loaders()

    model = ConvNet()
    model.to(device)

    optimizer = optim.SGD(model.parameters(),
                          lr=config["lr"], momentum=config["momentum"])
    
    for i in range(10):
        train(model, optimizer, train_loader, device=device)
        acc = test(model, test_loader, device=device)

        # When using WandbLoggerCallback, the metrics reported to tune are also logged in the W&B dashboard
        tune.report(mean_accuracy=acc)

        # @wandb_mixin enables logging custom metrics using wandb.log()
        error_rate = 100 * (1 - acc)
        wandb.log({"error_rate": error_rate})
```

## 🚀 Launching a Sweep with W&B and Ray/Tune

We're now almost ready to call `tune.run` to launch our hyperparameter sweep!
We just need to do three things:
1. set up a `wandb.Run`,
2. give the `WandbLoggerCallback` to `tune.run` so we can capture the output of `tune.report`, and
3. set up our hyperparameter sweep.

A `wandb.Run` is normally created by calling `wandb.init`.
`tune` will handle that for you, you just need to pass
the arguments as a dictionary
(see [our documentation](https://docs.wandb.com/library/init) for details on `wandb.init`).
At the bare minimum, you need to pass in a `project` name --
sort of like a `git` repo name, but for your ML projects.

In addition to holding arguments for `wandb.init`,
that dictionary also has a few special keys, described in
[the documentation for the `WandbLoggerCallback`](https://docs.ray.io/en/master/tune/tutorials/tune-wandb.html).


```
wandb.login()
```

We handle steps 2 and 3 when we invoke `tune.run`.

Step 2 is handled by passing in the `WandbLoggerCallback` class in a list
to the `loggers` argument of `tune.run`.

The setup of the hyperparameter sweep is handled by the
`config` argument of `tune.run`.
For the purposes of the integration,
the most important part is that this is where we pass in the `wandb_init`
dictionary.

This is also where we configure the "meat" of the hyperparameter sweep:
what are the hyperparameters we're sweeping over,
and how do we choose their values.

Here, we do a simple grid search, but
[Ray/Tune provides lots of sophisticated options](https://docs.ray.io/en/latest/tune/api_docs/suggestion.html).


```
analysis = tune.run(
    train_mnist,
    callbacks=[WandbLoggerCallback(project="raytune-colab")], # WandbLoggerCallback uses tune.run's logger interface
    resources_per_trial={"gpu": 1},
    config={
        # hyperparameters are set by keyword arguments
        "lr": tune.grid_search([0.0001, 0.001, 0.1]),
        "momentum": tune.grid_search([0.9, 0.99])
        }
    )

```


```
print("Best config: ", analysis.get_best_config(metric="mean_accuracy", mode="max"))
```
