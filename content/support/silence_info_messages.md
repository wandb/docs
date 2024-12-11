---
title: "How do I silence W&B info messages?"
toc_hide: true
type: docs
tags:
- notebooks
- environment variables
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

To turn off standard Weights & Biases logging and information messages, such as project info at the start of a run, set the `WANDB_SILENT` environment variable. This must occur in a notebook cell before running `wandb.login`:

<Tabs
  defaultValue="jupyter"
  values={[
    {label: 'Jupyter Magic', value: 'jupyter'},
    {label: 'Python', value: 'python'},
  ]}>
  <TabItem value="jupyter">

```notebook
%env WANDB_SILENT=True
```
  </TabItem>
  <TabItem value="python">

```python
import os

os.environ["WANDB_SILENT"] = "True"
```
  </TabItem>
</Tabs>

To suppress log messages such as `INFO SenderThread:11484 [sender.py:finish():979]` in your notebook, utilize the following code:

```python
import logging

logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)
```