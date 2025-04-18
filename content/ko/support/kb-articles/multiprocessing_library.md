---
title: Does W&B use the `multiprocessing` library?
menu:
  support:
    identifier: ko-support-kb-articles-multiprocessing_library
support:
- experiments
toc_hide: true
type: docs
url: /support/:filename
---

예, W&B는 `multiprocessing` 라이브러리를 사용합니다. 다음과 같은 오류 메시지는 가능한 문제를 나타냅니다.

```
An attempt has been made to start a new process before the current process 
has finished its bootstrapping phase.
```

이 문제를 해결하려면 `if __name__ == "__main__":` 를 사용하여 진입점 보호를 추가하십시오. 이 보호는 스크립트에서 직접 W&B를 실행할 때 필요합니다.
