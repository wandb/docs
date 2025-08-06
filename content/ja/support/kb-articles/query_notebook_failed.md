---
title: 「ノートブックのクエリに失敗しました」というエラーはどう対処すればよいですか？
url: /support/:filename
toc_hide: true
type: docs
support:
- ノートブック
- 環境変数
---

エラーメッセージ `"Failed to query for notebook name, you can set it manually with the WANDB_NOTEBOOK_NAME environment variable,"` が表示された場合は、環境変数を設定して解決できます。以下のいずれかの方法で設定してください。

{{< tabpane text=true >}}
{{% tab "Notebook" %}}
```python
# WANDB_NOTEBOOK_NAME 環境変数を設定
%env "WANDB_NOTEBOOK_NAME" "notebook name here"
```
{{% /tab %}}
{{% tab "Python" %}}
```python
import os

# WANDB_NOTEBOOK_NAME 環境変数を設定
os.environ["WANDB_NOTEBOOK_NAME"] = "notebook name here"
```
{{% /tab %}}
{{< /tabpane >}}