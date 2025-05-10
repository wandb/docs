---
title: How do I handle the 'Failed to query for notebook' error?
menu:
  support:
    identifier: ko-support-kb-articles-query_notebook_failed
support:
- notebooks
- environment variables
toc_hide: true
type: docs
url: /ko/support/:filename
translationKey: query_notebook_failed
---
`"Failed to query for notebook name, you can set it manually with the WANDB_NOTEBOOK_NAME environment variable,"` 오류 메시지가 발생하면 환경 변수를 설정하여 해결하십시오. 이를 수행하는 방법에는 여러 가지가 있습니다.

{{< tabpane text=true >}}
{{% tab "Notebook" %}}
```python
%env "WANDB_NOTEBOOK_NAME" "여기에 노트북 이름"
```
{{% /tab %}}
{{% tab "Python" %}}
```python
import os

os.environ["WANDB_NOTEBOOK_NAME"] = "여기에 노트북 이름"
```
{{% /tab %}}
{{< /tabpane >}}
