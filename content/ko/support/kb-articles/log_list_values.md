---
title: How do I log a list of values?
menu:
  support:
    identifier: ko-support-kb-articles-log_list_values
support:
- logs
- experiments
toc_hide: true
type: docs
url: /ko/support/:filename
translationKey: log_list_values
---
다음 예제는 [`wandb.log()`]({{< relref path="/ref/python/log/" lang="ko" >}})를 사용하여 여러 가지 방법으로 손실을 기록하는 방법을 보여줍니다.

{{< tabpane text=true >}}
{{% tab "사전 사용" %}}
```python
wandb.log({f"losses/loss-{ii}": loss for ii, 
  loss in enumerate(losses)})
```
{{% /tab %}}
{{% tab "히스토그램으로 사용" %}}
```python
# 손실을 히스토그램으로 변환합니다.
wandb.log({"losses": wandb.Histogram(losses)})  
```
{{% /tab %}}
{{< /tabpane >}}

자세한 내용은 [로깅에 대한 문서]({{< relref path="/guides/models/track/log/" lang="ko" >}})를 참조하세요.
