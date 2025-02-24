---
title: How do I handle the 'Failed to query for notebook' error?
menu:
  support:
    identifier: ko-support-query_notebook_failed
tags:
- notebooks
- environment variables
toc_hide: true
type: docs
---

`"Failed to query for notebook name, you can set it manually with the WANDB_NOTEBOOK_NAME environment variable,"` 오류 메시지가 발생하면 환경 변수를 설정하여 해결하세요. 이를 수행하는 방법은 여러 가지가 있습니다.

{{< tabpane text=true >}}
{{% tab "Notebook" %}}
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
