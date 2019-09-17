---
title: Using W&B Enterprise Cloud
sidebar_label: Usage
---

For the most part, you can use W&B Enterprise the same way as our Software-as-a-Service offering. There are a few tweaks to keep in mind.

## API Keys

W&B Enterprise API keys have a dedicated prefix to avoid confusion between the SaaS system and your private installation.

## Command line usage

Set the `WANDB_BASE_URL` environment variable to the host or IP address of your W&B server to point the wandb client in that direction.

```sh
WANDB_BASE_URL=http://1.2.3.4 python run train.py
```

## Jupyter notebooks

In jupyter notebooks, simply set the environment variable `WANDB_BASE_URL` to the host or IP address of the server running W&B enterprise. For example:

```python
import os
os.environ['WANDB_BASE_URL'] = 'http://1.2.3.4'

# At this point you will be asked for an API KEY to your server.
import wandb
wandb.init()
```
