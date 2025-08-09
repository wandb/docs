---
title: wandb.init 모드의 차이점은 무엇인가요?
menu:
  support:
    identifier: ko-support-kb-articles-difference_wandbinit_modes
support:
- Experiments
toc_hide: true
type: docs
url: /support/:filename
---

다음과 같은 모드들이 있습니다:

* `online` (기본값): 클라이언트가 데이터를 wandb 서버로 전송합니다.
* `offline`: 클라이언트가 데이터를 wandb 서버로 전송하지 않고, 로컬 머신에 저장합니다. 나중에 데이터를 동기화하려면 [`wandb sync`]({{< relref path="/ref/cli/wandb-sync.md" lang="ko" >}}) 코맨드를 사용하세요.
* `disabled`: 클라이언트가 실제로 동작하지 않고, 모의 오브젝트를 반환하며 네트워크 통신을 차단합니다. 모든 로그가 비활성화되지만, 모든 API 메소드 스텁은 여전히 호출 가능합니다. 주로 테스트 용도로 사용됩니다.