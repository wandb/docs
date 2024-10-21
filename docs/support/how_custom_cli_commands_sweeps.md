---
title: "How do I use custom CLI commands with sweeps?"
tags:
   - sweeps
---

You can use W&B Sweeps with custom CLI commands if you normally configure some aspects of training by passing command line arguments.

For example, the proceeding code snippet demonstrates a bash terminal where the user is training a Python script named train.py. The user passes in values that are then parsed within the Python script:

```bash
/usr/bin/env python train.py -b \
    your-training-config \
    --batchsize 8 \
    --lr 0.00001
```

To use custom commands, edit the `command` key in your YAML file. For example, continuing the example above, that might look like so:

```yaml
program:
  train.py
method: grid
parameters:
  batch_size:
    value: 8
  lr:
    value: 0.0001
command:
  - ${env}
  - python
  - ${program}
  - "-b"
  - your-training-config
  - ${args}
```

The `${args}` key expands to all the parameters in the sweep configuration file, expanded so they can be parsed by `argparse: --param1 value1 --param2 value2`

If you have extra arguments that you don't want to specify with `argparse` you can use:

```python
parser = argparse.ArgumentParser()
args, unknown = parser.parse_known_args()
```

:::info
Depending on the environment, `python` might point to Python 2. To ensure Python 3 is invoked, use `python3` instead of `python` when configuring the command:

```yaml
program:
  script.py
command:
  - ${env}
  - python3
  - ${program}
  - ${args}
```
:::