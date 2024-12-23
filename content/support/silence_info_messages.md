---
title: "How do I silence W&B info messages?"
toc_hide: true
type: docs
tags:
- notebooks
- environment variables
---
To suppress log messages in your notebook such as this:

```
INFO SenderThread:11484 [sender.py:finish():979]
``` 

Set the log level to `logging.ERROR` to only show errors, suppressing output of info-level log output.

```python
import logging

logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)
```

To turn off log output completely, set the `WANDB_SILENT` environment variable. This must occur in a notebook cell before running `wandb.login`:

{{< tabpane text=true langEqualsHeader=true >}}
{{% tab "Notebook" %}}
```python
%env WANDB_SILENT=True
```
{{% /tab %}}
{{% tab "Python" %}}
```python
import os

os.environ["WANDB_SILENT"] = "True"
```
{{% /tab %}}
{{< /tabpane >}}