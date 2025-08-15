---
title: '''Failed to query for notebook'' 오류는 어떻게 해결하나요?'
menu:
  support:
    identifier: ko-support-kb-articles-query_notebook_failed
support:
- 노트북
- 환경 변수
toc_hide: true
type: docs
url: /support/:filename
---

"Failed to query for notebook name, you can set it manually with the WANDB_NOTEBOOK_NAME environment variable,"라는 에러 메시지가 나타난다면 환경 변수를 직접 설정해서 해결할 수 있습니다. 다음과 같은 여러 방법이 있습니다:

{{< tabpane text=true >}}
{{% tab "Notebook" %}}
```python
# WANDB_NOTEBOOK_NAME 환경 변수를 직접 지정합니다.
%env "WANDB_NOTEBOOK_NAME" "notebook name here"
```
{{% /tab %}}
{{% tab "Python" %}}
```python
import os

# WANDB_NOTEBOOK_NAME 환경 변수를 직접 지정합니다.
os.environ["WANDB_NOTEBOOK_NAME"] = "notebook name here"
```
{{% /tab %}}
{{< /tabpane >}}