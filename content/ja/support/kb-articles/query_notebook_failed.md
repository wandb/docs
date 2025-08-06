---
title: 「Failed to query for notebook」エラーの対処方法は？
menu:
  support:
    identifier: ja-support-kb-articles-query_notebook_failed
support:
- ノートブック
- 環境変数
toc_hide: true
type: docs
url: /support/:filename
---

"Failed to query for notebook name, you can set it manually with the WANDB_NOTEBOOK_NAME environment variable," というエラーメッセージが表示された場合は、環境変数を設定することで解決できます。以下のいくつかの方法で設定が可能です。

{{< tabpane text=true >}}
{{% tab "ノートブック" %}}
```python
# WANDB_NOTEBOOK_NAME 環境変数を設定します
%env "WANDB_NOTEBOOK_NAME" "notebook name here"
```
{{% /tab %}}
{{% tab "Python" %}}
```python
import os

# WANDB_NOTEBOOK_NAME 環境変数を指定します
os.environ["WANDB_NOTEBOOK_NAME"] = "notebook name here"
```
{{% /tab %}}
{{< /tabpane >}}