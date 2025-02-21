---
title: How do I stop wandb from writing to my terminal or my Jupyter notebook output?
menu:
  support:
    identifier: ja-support-stop_wandb_writing_terminal_jupyter_notebook_output
tags:
- environment variables
toc_hide: true
type: docs
---

環境変数 [`WANDB_SILENT`]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) を `true` に設定します。

{{< tabpane text=true langEqualsHeader=true >}}
  {{% tab header="Python" %}}
```python
os.environ["WANDB_SILENT"] = "true"
```
  {{% /tab %}}
  {{% tab "Notebook" %}}
```python
%env WANDB_SILENT=true
```
  {{% /tab %}}
  {{% tab "Command-Line" %}}
```shell
WANDB_SILENT=true
```
  {{% /tab %}}
{{< /tabpane >}}
