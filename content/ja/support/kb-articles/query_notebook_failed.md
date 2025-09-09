---
title: 「Failed to query for notebook」 エラーはどう対処すればよいですか？
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

エラーメッセージ "Failed to query for notebook name, you can set it manually with the WANDB_NOTEBOOK_NAME environment variable," が表示された場合は、環境変数を設定して解決してください。複数の方法があります:

{{< tabpane text=true >}}
{{% tab "ノートブック" %}}
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