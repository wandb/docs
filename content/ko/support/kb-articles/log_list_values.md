---
title: 리스트 형태의 값을 어떻게 로그할 수 있나요?
menu:
  support:
    identifier: ko-support-kb-articles-log_list_values
support:
- 로그
- 실험
toc_hide: true
type: docs
url: /support/:filename
---

이 예제들은 [`wandb.Run.log()`]({{< relref path="/ref/python/sdk/classes/run/#method-runlog/" lang="ko" >}})을 사용하여 다양한 방식으로 loss를 로그하는 방법을 보여줍니다.

{{< tabpane text=true >}}
{{% tab "딕셔너리 사용" %}}
```python
import wandb

# 새로운 run을 초기화
with wandb.init(project="log-list-values", name="log-dict") as run:
    # 딕셔너리로 loss 값들을 로그
    losses = [0.1, 0.2, 0.3, 0.4, 0.5]
    run.log({"losses": losses})
    run.log({f"losses/loss-{ii}": loss for ii, loss in enumerate(losses)})
```
{{% /tab %}}
{{% tab "히스토그램으로 저장" %}}
```python
import wandb

# 새로운 run을 초기화
with wandb.init(project="log-list-values", name="log-hist") as run:
    # loss 값들을 히스토그램으로 로그
    losses = [0.1, 0.2, 0.3, 0.4, 0.5]
    run.log({"losses": wandb.Histogram(losses)})
```
{{% /tab %}}
{{< /tabpane >}}

더 많은 내용을 보려면 [로그 관련 문서]({{< relref path="/guides/models/track/log/" lang="ko" >}})를 참고하세요.