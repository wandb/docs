---
title: 누가 내 Artifacts에 접근할 수 있나요?
menu:
  support:
    identifier: ko-support-kb-articles-access_artifacts
support:
- Artifacts
toc_hide: true
type: docs
url: /support/:filename
---

Artifacts 는 상위 Project 의 엑세스 권한을 상속받습니다.

* Private Project 에서는 팀 멤버만 Artifacts 에 엑세스할 수 있습니다.
* Public Project 에서는 모든 사용자가 Artifacts 를 읽을 수 있지만, 팀 멤버만 생성하거나 수정할 수 있습니다.
* Open Project 에서는 모든 사용자가 Artifacts 를 읽고 쓸 수 있습니다.

## Artifacts 워크플로우

이 섹션에서는 Artifacts 를 관리하고 편집하는 워크플로우를 안내합니다. 많은 워크플로우에서 [W&B API]({{< relref path="/guides/models/track/public-api-guide.md" lang="ko" >}})를 활용하며, 이는 [클라이언트 라이브러리]({{< relref path="/ref/python/" lang="ko" >}})의 한 부분으로 W&B 에 저장된 데이터에 엑세스할 수 있게 해줍니다.