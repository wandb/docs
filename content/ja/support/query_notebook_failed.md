---
title: How do I handle the 'Failed to query for notebook' error?
menu:
  support:
    identifier: ja-support-query_notebook_failed
tags:
- notebooks
- environment variables
toc_hide: true
type: docs
---

エラーメッセージ `"Failed to query for notebook name, you can set it manually with the WANDB_NOTEBOOK_NAME environment variable,"` に遭遇した場合は、環境変数を設定することで解決できます。これを達成する方法はいくつかあります:

{{< tabpane text=true >}}
{{% tab "Notebook" %}}
```python
%env "WANDB_NOTEBOOK_NAME" "ノートブック名をここに"
```
{{% /tab %}}
{{% tab "Python" %}}
```python
import os

os.environ["WANDB_NOTEBOOK_NAME"] = "ノートブック名をここに"
```
{{% /tab %}}
{{< /tabpane >}}