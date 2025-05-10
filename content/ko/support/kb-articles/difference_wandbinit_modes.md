---
title: What is the difference between wandb.init modes?
menu:
  support:
    identifier: ko-support-kb-articles-difference_wandbinit_modes
support:
- experiments
toc_hide: true
type: docs
url: /ko/support/:filename
translationKey: difference_wandbinit_modes
---
다음 모드를 사용할 수 있습니다.

* `online` (기본값): 클라이언트가 데이터를 wandb 서버로 보냅니다.
* `offline`: 클라이언트가 데이터를 wandb 서버로 보내는 대신 로컬 머신에 데이터를 저장합니다. 나중에 데이터를 동기화하려면 [`wandb sync`]({{< relref path="/ref/cli/wandb-sync.md" lang="ko" >}}) 코맨드를 사용하세요.
* `disabled`: 클라이언트가 모의 오브젝트를 반환하여 작업을 시뮬레이션하고 네트워크 통신을 방지합니다. 모든 로깅이 꺼져 있지만 모든 API 메소드 스텁은 호출 가능한 상태로 유지됩니다. 이 모드는 일반적으로 테스트에 사용됩니다.
