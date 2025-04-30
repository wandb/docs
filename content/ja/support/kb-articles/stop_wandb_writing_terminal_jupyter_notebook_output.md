---
title: 私のターミナルまたは Jupyter ノートブック出力に書き込むのを wandb にやめさせるにはどうすればいいですか？
menu:
  support:
    identifier: ja-support-kb-articles-stop_wandb_writing_terminal_jupyter_notebook_output
support:
  - environment variables
toc_hide: true
type: docs
url: /ja/support/:filename
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