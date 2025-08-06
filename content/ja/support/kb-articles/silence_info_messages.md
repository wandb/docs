---
title: W&B の INFO メッセージを非表示にするにはどうすればよいですか？
url: /support/:filename
toc_hide: true
type: docs
support:
- ノートブック
- 環境変数
---

ノートブック内で以下のようなログメッセージを非表示にするには:

```
INFO SenderThread:11484 [sender.py:finish():979]
```

ログレベルを `logging.ERROR` に設定することで、エラーメッセージのみを表示し、INFO レベルのログ出力を抑制できます。

```python
import logging

logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)
```

ログ出力をさらに減らしたい場合は、`WANDB_QUIET` 環境変数を `True` に設定してください。ログ出力を完全にオフにしたい場合は、`WANDB_SILENT` 環境変数を `True` に設定します。ノートブックでは、`wandb.login` を実行する前に `WANDB_QUIET` または `WANDB_SILENT` を設定してください。

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