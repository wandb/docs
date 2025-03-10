---
title: "How do I use custom CLI commands with sweeps?"
toc_hide: true
type: docs
tags:
   - sweeps
---
You can use W&B Sweeps with custom CLI commands if training configuration passes command-line arguments.

In the example below, the code snippet illustrates a bash terminal where a user trains a Python script named `train.py`, providing values that the script parses:

```bash
/usr/bin/env python train.py -b \
    your-training-config \
    --batchsize 8 \
    --lr 0.00001
```

To implement custom commands, modify the `command` key in the YAML file. Based on the previous example, the configuration appears as follows:

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

The `${args}` key expands to all parameters in the sweep configuration, formatted for `argparse` as `--param1 value1 --param2 value2`.

For additional arguments outside of `argparse`, implement the following:

```python
parser = argparse.ArgumentParser()
args, unknown = parser.parse_known_args()
```

{{% alert %}}
Depending on the environment, `python` might refer to Python 2. To ensure invocation of Python 3, use `python3` in the command configuration:

```yaml
program:
  script.py
command:
  - ${env}
  - python3
  - ${program}
  - ${args}
```
{{% /alert %}}