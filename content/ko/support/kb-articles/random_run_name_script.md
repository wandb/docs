---
title: 스크립트에서 무작위 run 이름을 어떻게 얻을 수 있나요?
menu:
  support:
    identifier: ko-support-kb-articles-random_run_name_script
support:
- 실험
toc_hide: true
type: docs
url: /support/:filename
---

run 오브젝트의 `.save()` 메소드를 호출하여 현재 run 을 저장하세요. run 오브젝트의 `name` 속성을 사용하여 이름을 가져올 수 있습니다.