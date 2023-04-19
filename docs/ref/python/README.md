# Python Library




[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)View source on GitHub](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/__init__.py)



Use wandb to track machine learning work.


The most commonly used functions/objects are:
 - wandb.init — initialize a new run at the top of your training script
 - wandb.config — track hyperparameters and metadata
 - wandb.log — log metrics and media over time within your training loop

For guides and examples, see https://docs.wandb.ai.

For scripts and interactive notebooks, see https://github.com/wandb/examples.

For reference documentation, see https://docs.wandb.com/ref/python.

## Classes

[`class Artifact`](./artifact.md): Flexible and lightweight building block for dataset and model versioning.

[`class Run`](./run.md): A unit of computation logged by wandb. Typically, this is an ML experiment.

## Functions

[`agent(...)`](./agent.md): Run a function or program with configuration parameters specified by server.

[`controller(...)`](./controller.md): Public sweep controller constructor.

[`finish(...)`](./finish.md): Mark a run as finished, and finish uploading all data.

[`init(...)`](./init.md): Start a new run to track and log to W&B.

[`log(...)`](./log.md): Log a dictonary of data to the current run's history.

[`save(...)`](./save.md): Ensure all files matching `glob_str` are synced to wandb with the policy specified.

[`sweep(...)`](./sweep.md): Initialize a hyperparameter sweep.

[`watch(...)`](./watch.md): Hook into the torch model to collect gradients and the topology.



| Other Members | |
| :--- | :--- |
| `__version__` | `'0.13.11'` |
| `config` | |
| `summary` | |

