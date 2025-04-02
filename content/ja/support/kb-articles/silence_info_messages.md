---
title: How do I silence W&B info messages?
menu:
  support:
    identifier: ja-support-kb-articles-silence_info_messages
support:
- notebooks
- environment variables
toc_hide: true
type: docs
url: /support/:filename
---

あなたの notebook で、以下のようなログメッセージを抑制するには:

```
INFO SenderThread:11484 [sender.py:finish():979]
```

ログレベルを `logging.ERROR` に設定して、エラーのみを表示し、info レベルのログ出力を抑制します。

```python
import logging

logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)
```

ログ出力を完全にオフにするには、`WANDB_SILENT` 環境 変数 を設定します。これは、`wandb.login` を実行する前に notebook のセルで設定する必要があります。

{{< tabpane text=true langEqualsHeader=true >}}
{{% tab "Notebook" %}}
```python
%env WANDB_SILENT=True
```
{{% /tab %}}
{{% tab "Python" %}}
```python
import os

os.environ["WANDB_SILENT"] = "True"
```
{{% /tab %}}
{{< /tabpane >}}
