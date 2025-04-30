---
title: 'InitStartError: Error communicating with wandb process'
menu:
  support:
    identifier: ko-support-kb-articles-initstarterror_error_communicating_wandb_process
support:
- experiments
toc_hide: true
type: docs
url: /ko/support/:filename
---

이 오류는 라이브러리가 데이터를 서버와 동기화하는 프로세스를 시작하는 데 문제가 발생했음을 나타냅니다.

다음 해결 방법은 특정 환경에서 문제를 해결합니다.

{{< tabpane text=true >}}
{{% tab "Linux and OS X" %}}
```python
wandb.init(settings=wandb.Settings(start_method="fork"))
```

{{% /tab %}}
{{% tab "Google Colab" %}}

`0.13.0` 이전 버전의 경우 다음을 사용하세요.

```python
wandb.init(settings=wandb.Settings(start_method="thread"))
```
{{% /tab %}}
{{< /tabpane >}}
