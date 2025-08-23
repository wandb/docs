---
title: wandb를 멀티프로세싱, 예를 들어 분산 트레이닝에서 어떻게 사용할 수 있나요?
menu:
  support:
    identifier: ko-support-kb-articles-multiprocessing_eg_distributed_training
support:
- 실험
toc_hide: true
type: docs
url: /support/:filename
---

트레이닝 프로그램이 여러 프로세스를 사용할 경우, `wandb.init()`이 호출되지 않은 프로세스에서 wandb 메소드를 호출하지 않도록 프로그램을 구성하세요.

멀티프로세스 트레이닝을 관리하는 방법은 다음과 같습니다.

1. 모든 프로세스에서 `wandb.init`을 호출하고, [group]({{< relref path="/guides/models/track/runs/grouping.md" lang="ko" >}}) 키워드 인수를 사용하여 공통 그룹을 만드세요. 각 프로세스는 별도의 wandb run을 가지며, UI에서는 이 트레이닝 프로세스들을 함께 그룹으로 보여줍니다.
2. 한 개의 프로세스에서만 `wandb.init`을 호출하고, [multiprocessing 큐](https://docs.python.org/3/library/multiprocessing.html#exchanging-objects-between-processes)를 통해 로그할 데이터를 전달하세요.

{{% alert %}}
이러한 접근 방식에 대한 자세한 설명과 Torch DDP 예제가 포함된 코드는 [분산 트레이닝 가이드]({{< relref path="/guides/models/track/log/distributed-training.md" lang="ko" >}})를 참고하세요.
{{% /alert %}}