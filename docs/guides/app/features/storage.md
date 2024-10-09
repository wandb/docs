---
title: Manage storage
description: W&B 데이터 저장소 관리 방법.
displayed_sidebar: default
---

저장 한도에 접근하거나 초과하고 있다면, 데이터를 관리하기 위한 여러 경로가 있습니다. 어떤 경로가 가장 적합한지는 계정 유형과 현재 프로젝트 설정에 따라 다를 것입니다.

## 저장소 사용량 관리
W&B는 저장소 사용량을 최적화하기 위한 다양한 메소드를 제공합니다:

- W&B 스토리지에 업로드하는 대신, W&B 시스템 외부에 저장된 파일을 추적하기 위해 [reference artifacts](../../artifacts/track-external-files.md)를 사용하십시오.
- 저장소를 위한 [외부 클라우드 스토리지 버킷](../features/teams.md)을 사용하십시오. *(기업 전용)*

## 데이터 삭제
저장 한도 내에 머물기 위해 데이터를 삭제하는 것도 선택할 수 있습니다. 여러 가지 방법으로 이를 수행할 수 있습니다:

- 앱 UI를 사용하여 데이터를 인터랙티브하게 삭제합니다.
- Artifacts에 [TTL 정책을 설정](../../artifacts/ttl.md)하여 자동으로 삭제되도록 합니다.