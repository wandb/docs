---
title: 'InitStartError: wandb 프로세스와 통신 중 오류가 발생했습니다'
menu:
  support:
    identifier: ko-support-kb-articles-initstarterror_error_communicating_wandb_process
support:
- 실험
toc_hide: true
type: docs
url: /support/:filename
---

이 오류는 라이브러리가 데이터 를 서버로 동기화하는 프로세스 를 실행하는 데 문제가 발생했음을 나타냅니다.

아래의 해결 방법들은 특정 환경 에서 이 문제를 해결할 수 있습니다:

{{< tabpane text=true >}}
{{% tab "Linux 및 OS X" %}}
```python
wandb.init(settings=wandb.Settings(start_method="fork"))
```

{{% /tab %}}
{{% tab "Google Colab" %}}

`0.13.0` 버전 이전의 경우, 다음 코드를 사용하세요:

```python
wandb.init(settings=wandb.Settings(start_method="thread"))
```
{{% /tab %}}
{{< /tabpane >}}