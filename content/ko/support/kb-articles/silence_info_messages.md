---
title: W&B 안내 메시지를 어떻게 끌 수 있나요?
menu:
  support:
    identifier: ko-support-kb-articles-silence_info_messages
support:
- 노트북
- 환경 변수
toc_hide: true
type: docs
url: /support/:filename
---

노트북에서 다음과 같은 로그 메시지를 숨기려면:

```
INFO SenderThread:11484 [sender.py:finish():979]
```

로그 레벨을 `logging.ERROR` 로 설정하면 에러만 표시되고, info 수준의 로그 출력은 숨길 수 있습니다.

```python
import logging

logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)
```

로그 출력을 크게 줄이고 싶다면 `WANDB_QUIET` 환경 변수를 `True` 로 설정하세요. 로그 출력을 아예 끄고 싶다면 `WANDB_SILENT` 환경 변수를 `True` 로 설정하면 됩니다. 노트북에서는 `wandb.login` 을 실행하기 전에 `WANDB_QUIET` 또는 `WANDB_SILENT` 를 먼저 설정하세요:

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