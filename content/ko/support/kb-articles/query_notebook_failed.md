---
menu:
  support:
    identifier: ko-support-kb-articles-query_notebook_failed
support:
- notebooks
- environment variables
title: How do I handle the 'Failed to query for notebook' error?
toc_hide: true
type: docs
url: /support/:filename
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