---
title: W&B에서 `multiprocessing` 라이브러리를 사용하나요?
menu:
  support:
    identifier: ko-support-kb-articles-multiprocessing_library
support:
- 실험
toc_hide: true
type: docs
url: /support/:filename
---

네, W&B는 `multiprocessing` 라이브러리를 사용합니다. 다음과 같은 오류 메시지는 문제가 있을 수 있음을 나타냅니다:

```
현재 프로세스가 부트스트랩 단계가 끝나기 전에 새 프로세스를 시작하려고 했습니다.
```

이 문제를 해결하려면, `if __name__ == "__main__":`로 엔트리 포인트 보호 구문을 추가하세요. 이 보호 구문은 W&B를 스크립트에서 직접 실행할 때 필요합니다.