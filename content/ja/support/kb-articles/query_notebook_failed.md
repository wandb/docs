---
title: ノートブックのクエリに失敗しましたというエラーはどのように対処しますか？
menu:
  support:
    identifier: ja-support-kb-articles-query_notebook_failed
support:
  - notebooks
  - environment variables
toc_hide: true
type: docs
url: /ja/support/:filename
translationKey: query_notebook_failed
---
`"Failed to query for notebook name, you can set it manually with the WANDB_NOTEBOOK_NAME environment variable,"` というエラーメッセージが表示された場合は、環境変数を設定することで解決できます。これを達成するための複数のメソッドがあります:

{{< tabpane text=true >}}
{{% tab "Notebook" %}}
```python
%env "WANDB_NOTEBOOK_NAME" "ノートブック名はこちら"
```
{{% /tab %}}
{{% tab "Python" %}}
```python
import os

os.environ["WANDB_NOTEBOOK_NAME"] = "ノートブック名はこちら"
```
{{% /tab %}}
{{< /tabpane >}}