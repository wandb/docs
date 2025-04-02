---
title: How do I silence W&B info messages?
menu:
  support:
    identifier: ko-support-kb-articles-silence_info_messages
support:
- notebooks
- environment variables
toc_hide: true
type: docs
url: /support/:filename
---

다음과 같이 노트북에서 로그 메시지를 표시하지 않으려면:

```
INFO SenderThread:11484 [sender.py:finish():979]
```

로그 수준을 `logging.ERROR`로 설정하여 오류만 표시하고 정보 수준 로그 출력을 표시하지 않도록 합니다.

```python
import logging

logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)
```

로그 출력을 완전히 끄려면 `WANDB_SILENT` 환경 변수를 설정합니다. `wandb.login`을 실행하기 전에 노트북 셀에서 이 작업을 수행해야 합니다.

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
