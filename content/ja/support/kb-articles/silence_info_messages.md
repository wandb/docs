---
title: W&B の情報メッセージを消音するにはどうすればいいですか？
menu:
  support:
    identifier: ja-support-kb-articles-silence_info_messages
support:
- ノートブック
- 環境変数
toc_hide: true
type: docs
url: /support/:filename
---

ノートブックで次のようなログメッセージを抑制するには:

```
INFO SenderThread:11484 [sender.py:finish():979]
```

ログレベルを `logging.ERROR` に設定してエラーのみを表示し、情報レベルのログ出力を抑制します。

```python
import logging

logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)
```

ログ出力を完全にオフにするには、`WANDB_SILENT` 環境変数を設定します。これは `wandb.login` を実行する前にノートブックセル内で行う必要があります。

{{< tabpane text=true langEqualsHeader=true >}}
{{% tab "ノートブック" %}}
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