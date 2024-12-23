---
title: "How do I stop wandb from writing to my terminal or my Jupyter notebook output?"
toc_hide: true
type: docs
tags:
   - environment variables
---
Set the environment variable [`WANDB_SILENT`](../guides/track/environment-variables.md) to `true`.

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