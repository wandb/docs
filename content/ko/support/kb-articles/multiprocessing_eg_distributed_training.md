---
title: How can I use wandb with multiprocessing, e.g. distributed training?
menu:
  support:
    identifier: ko-support-kb-articles-multiprocessing_eg_distributed_training
support:
- experiments
toc_hide: true
type: docs
url: /support/:filename
---

만약 트레이닝 프로그램이 여러 프로세스를 사용하는 경우, `wandb.init()` 없이 프로세스에서 wandb 메소드 호출을 하지 않도록 프로그램을 구성하세요.

다음과 같은 방법으로 멀티프로세스 트레이닝을 관리합니다:

1. 모든 프로세스에서 `wandb.init`을 호출하고 [group]({{< relref path="/guides/models/track/runs/grouping.md" lang="ko" >}}) 키워드 인수를 사용하여 공유 그룹을 생성합니다. 각 프로세스는 자체 wandb run을 가지며, UI는 트레이닝 프로세스를 함께 그룹화합니다.
2. 하나의 프로세스에서만 `wandb.init`을 호출하고 [multiprocessing queues](https://docs.python.org/3/library/multiprocessing.html#exchanging-objects-between-processes) 를 통해 로그할 데이터를 전달합니다.

{{% alert %}}
Torch DDP를 사용한 코드 예제를 포함하여 이러한 접근 방식에 대한 자세한 설명은 [Distributed Training Guide]({{< relref path="/guides/models/track/log/distributed-training.md" lang="ko" >}}) 를 참조하십시오.
{{% /alert %}}
