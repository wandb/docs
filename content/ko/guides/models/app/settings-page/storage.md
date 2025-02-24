---
title: Manage storage
description: W&B 데이터 저장소를 관리하는 방법.
menu:
  default:
    identifier: ko-guides-models-app-settings-page-storage
    parent: settings
weight: 60
---

스토리지 제한에 거의 도달했거나 초과한 경우, 데이터를 관리하기 위한 여러 가지 방법이 있습니다. 가장 적합한 방법은 계정 유형과 현재 프로젝트 설정에 따라 달라집니다.

## 스토리지 소비량 관리
W&B는 스토리지 소비량을 최적화하는 다양한 방법을 제공합니다.

- [레퍼런스 Artifacts]({{< relref path="/guides/core/artifacts/track-external-files.md" lang="ko" >}})를 사용하여 W&B 시스템 외부에서 저장된 파일을 추적합니다. W&B 스토리지에 업로드하는 대신에 말이죠.
- 스토리지를 위해 [외부 클라우드 스토리지 버킷]({{< relref path="teams.md" lang="ko" >}})을 사용합니다. *(Enterprise 전용)*

## 데이터 삭제
스토리지 제한을 초과하지 않기 위해 데이터를 삭제할 수도 있습니다. 이를 수행하는 방법은 여러 가지가 있습니다.

- 앱 UI를 사용하여 대화형으로 데이터를 삭제합니다.
- Artifacts에 [TTL 정책 설정]({{< relref path="/guides/core/artifacts/manage-data/ttl.md" lang="ko" >}})하여 자동으로 삭제되도록 합니다.
