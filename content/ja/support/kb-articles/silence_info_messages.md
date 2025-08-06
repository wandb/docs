---
title: W&B の情報メッセージを非表示にするにはどうすればいいですか？
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

ノートブックで以下のようなログメッセージを非表示にするには:

```
INFO SenderThread:11484 [sender.py:finish():979]
```

`logging.ERROR` にログレベルを設定すると、エラーのみが表示され、info レベルのログ出力が抑制されます。

```python
import logging

logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)
```

ログ出力を大幅に減らすには、`WANDB_QUIET` 環境変数を `True` に設定してください。ログ出力を完全にオフにしたい場合は、`WANDB_SILENT` 環境変数を `True` に設定します。ノートブックで使用する場合は、`wandb.login` を実行する前に `WANDB_QUIET` または `WANDB_SILENT` を設定してください。

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