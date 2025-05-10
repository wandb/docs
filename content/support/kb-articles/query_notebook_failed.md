---
url: /support/:filename
title: "How do I handle the 'Failed to query for notebook' error?"
toc_hide: true
type: docs
support:
- notebooks
- environment variables
translationKey: query_notebook_failed
---
If you encounter the error message `"Failed to query for notebook name, you can set it manually with the WANDB_NOTEBOOK_NAME environment variable,"` resolve it by setting the environment variable. Multiple methods accomplish this:

{{< tabpane text=true >}}
{{% tab "Notebook" %}}
```python
%env "WANDB_NOTEBOOK_NAME" "notebook name here"
```
{{% /tab %}}
{{% tab "Python" %}}
```python
import os

os.environ["WANDB_NOTEBOOK_NAME"] = "notebook name here"
```
{{% /tab %}}
{{< /tabpane >}}