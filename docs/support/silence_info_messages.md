---
title: How do I silence W&B info messages?
tags:
- notebooks
- environment variables
---
To turn off standard wandb logging and info messages (e.g. project info at the start of a run), run the following in a notebook cell _before_ running `wandb.login` to set the `WANDB_SILENT` environment variable:

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

If you see log messages like `INFO SenderThread:11484 [sender.py:finish():979]` in your notebook, you can turn those off with the following:

```python
import logging

logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)
```
