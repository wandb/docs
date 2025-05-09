---
title: Manage data governance and access control
description: 모델 레지스트리 역할 기반 엑세스 제어(RBAC)를 사용하여 보호된 에일리어스를 업데이트할 수 있는 사람을 제어합니다.
menu:
  default:
    identifier: ko-guides-core-registry-model_registry-access_controls
    parent: model-registry
weight: 10
---

*보호된 에일리어스*를 사용하여 모델 개발 파이프라인의 주요 단계를 나타냅니다. *모델 레지스트리 관리자* 만이 보호된 에일리어스를 추가, 수정 또는 제거할 수 있습니다. 모델 레지스트리 관리자는 보호된 에일리어스를 정의하고 사용할 수 있습니다. W&B는 관리자가 아닌 사용자가 모델 버전에서 보호된 에일리어스를 추가하거나 제거하는 것을 차단합니다.

{{% alert %}}
팀 관리자 또는 현재 레지스트리 관리자만이 레지스트리 관리자 목록을 관리할 수 있습니다.
{{% /alert %}}

예를 들어, `staging` 및 `production`을 보호된 에일리어스로 설정했다고 가정합니다. 팀의 모든 구성원은 새로운 모델 버전을 추가할 수 있습니다. 그러나 관리자만이 `staging` 또는 `production` 에일리어스를 추가할 수 있습니다.

## 엑세스 제어 설정
다음 단계에서는 팀의 모델 레지스트리에 대한 엑세스 제어를 설정하는 방법을 설명합니다.

1. [https://wandb.ai/registry/model](https://wandb.ai/registry/model)의 W&B Model Registry 앱으로 이동합니다.
2. 페이지 오른쪽 상단의 톱니바퀴 버튼을 선택합니다.
{{< img src="/images/models/rbac_gear_button.png" alt="" >}}
3. **레지스트리 관리자 관리** 버튼을 선택합니다.
4. **멤버** 탭에서 모델 버전에서 보호된 에일리어스를 추가하고 제거할 수 있는 엑세스 권한을 부여할 사용자를 선택합니다.
{{< img src="/images/models/access_controls_admins.gif" alt="" >}}

## 보호된 에일리어스 추가
1. [https://wandb.ai/registry/model](https://wandb.ai/registry/model)의 W&B Model Registry 앱으로 이동합니다.
2. 페이지 오른쪽 상단의 톱니바퀴 버튼을 선택합니다.
{{< img src="/images/models/rbac_gear_button.png" alt="" >}}
3. **보호된 에일리어스** 섹션으로 스크롤합니다.
4. 더하기 아이콘(**+**) 아이콘을 클릭하여 새 에일리어스를 추가합니다.
{{< img src="/images/models/access_controls_add_protected_aliases.gif" alt="" >}}
