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

`"Failed to query for notebook name, you can set it manually with the WANDB_NOTEBOOK_NAME environment variable,"` というエラーメッセージが表示された場合は、 環境 変数を設定して解決してください。これを実現する方法は複数あります。

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
