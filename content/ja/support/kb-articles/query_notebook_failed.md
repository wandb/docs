---
title: How do I handle the 'Failed to query for notebook' error?
menu:
  support:
    identifier: ja-support-kb-articles-query_notebook_failed
support:
- notebooks
- environment variables
toc_hide: true
type: docs
url: /support/:filename
---

`"Failed to query for notebook name, you can set it manually with the WANDB_NOTEBOOK_NAME 環境 変数,"` というエラーメッセージが表示された場合は、 環境 変数を設定して解決してください。これには複数の メソッド があります。

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
