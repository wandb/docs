---
title: Who has access to my artifacts?
menu:
  support:
    identifier: ko-support-access_artifacts
tags:
- artifacts
toc_hide: true
type: docs
---

Artifacts는 상위 프로젝트로부터 엑세스 권한을 상속받습니다.

* 비공개 프로젝트에서는 팀 멤버만이 Artifacts에 엑세스할 수 있습니다.
* 공개 프로젝트에서는 모든 사용자가 Artifacts를 읽을 수 있지만, 팀 멤버만이 Artifacts를 생성하거나 수정할 수 있습니다.
* 공개된 프로젝트에서는 모든 사용자가 Artifacts를 읽고 쓸 수 있습니다.

## Artifacts 워크플로우

이 섹션에서는 Artifacts를 관리하고 편집하기 위한 워크플로우를 설명합니다. 많은 워크플로우가 W&B에 저장된 데이터에 엑세스를 제공하는 [클라이언트 라이브러리]({{< relref path="/ref/python/" lang="ko" >}})의 구성 요소인 [W&B API]({{< relref path="/guides/models/track/public-api-guide.md" lang="ko" >}})를 활용합니다.
