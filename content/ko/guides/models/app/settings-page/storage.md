---
title: 스토리지 관리
description: W&B 데이터 저장소를 관리하는 방법.
menu:
  default:
    identifier: ko-guides-models-app-settings-page-storage
    parent: settings
weight: 60
---

저장 한도에 근접하거나 초과하는 경우, 데이터를 관리할 수 있는 여러 가지 방법이 있습니다. 어떤 방법이 가장 적합한지는 계정 유형과 현재 프로젝트 설정에 따라 다를 수 있습니다.

## 저장소 사용량 관리
W&B는 저장소 사용 최적화를 위한 다양한 방법을 제공합니다.

- [reference artifacts]({{< relref path="/guides/core/artifacts/track-external-files.md" lang="ko" >}})를 사용하여, W&B 시스템 외부에 저장된 파일을 업로드하지 않고 추적할 수 있습니다.
- [external cloud storage bucket]({{< relref path="teams.md" lang="ko" >}})을 저장소로 사용할 수 있습니다. *(Enterprise 전용)*

## 데이터 삭제
저장 한도를 초과하지 않도록 데이터를 삭제할 수도 있습니다. 다음과 같은 방법이 있습니다.

- 앱 UI를 사용해 데이터를 직접 삭제할 수 있습니다.
- Artifacts에 [TTL 정책]({{< relref path="/guides/core/artifacts/manage-data/ttl.md" lang="ko" >}})을 설정해 자동으로 삭제되도록 할 수 있습니다.