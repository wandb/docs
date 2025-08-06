---
title: wandb が ターミナル や Jupyter ノートブック の出力に書き込むのを止めるにはどうすればいいですか？
url: /support/:filename
toc_hide: true
type: docs
support:
- 環境変数
---

環境変数 [`WANDB_SILENT`]({{< relref "/guides/models/track/environment-variables.md" >}}) を `true` に設定します。

{{< tabpane text=true langEqualsHeader=true >}}
  {{% tab header="Python" %}}
```python
# 環境変数を設定します
os.environ["WANDB_SILENT"] = "true"
```
  {{% /tab %}}
  {{% tab "Notebook" %}}
```python
# ノートブック環境での設定方法
%env WANDB_SILENT=true
```
  {{% /tab %}}
  {{% tab "Command-Line" %}}
```shell
# コマンドラインでの設定方法
WANDB_SILENT=true
```
  {{% /tab %}}
{{< /tabpane >}}