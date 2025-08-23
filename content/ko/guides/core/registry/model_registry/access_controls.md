---
title: 데이터 거버넌스 및 엑세스 제어 관리
description: 모델 레지스트리 역할 기반 엑세스 제어(RBAC)를 사용하여 보호된 에일리어스 를 누가 업데이트할 수 있는지 제어할 수 있습니다.
menu:
  default:
    identifier: ko-guides-core-registry-model_registry-access_controls
    parent: model-registry
weight: 10
---

*Protected alias*를 사용하여 모델 개발 파이프라인의 핵심 단계를 표현하세요. *Model Registry Administrator*만이 protected alias를 추가, 수정 또는 삭제할 수 있습니다. Model registry 관리자는 protected alias를 정의하고 사용할 수 있습니다. W&B는 관리자가 아닌 사용자가 모델 버전에서 protected alias를 추가하거나 제거하는 것을 차단합니다.

{{% alert %}}
Team 관리자 또는 현재 registry 관리자인 경우에만 registry 관리자 목록을 관리할 수 있습니다.
{{% /alert %}}

예를 들어, `staging`과 `production`을 protected alias로 설정했다고 가정해봅시다. 팀의 모든 멤버는 새로운 모델 버전을 추가할 수 있지만, 관리자만이 `staging` 또는 `production` 에일리어스를 추가할 수 있습니다.


## 엑세스 제어 설정하기
다음 단계에서는 팀의 model registry에 대한 엑세스 제어를 설정하는 방법을 설명합니다.

1. [W&B Model Registry 앱](https://wandb.ai/registry/model)으로 이동하세요.
2. 페이지 오른쪽 상단에 있는 톱니바퀴 버튼을 선택하세요.
{{< img src="/images/models/rbac_gear_button.png" alt="Registry settings gear" >}}
3. **Manage registry admins** 버튼을 선택하세요.
4. **Members** 탭에서, 모델 버전에서 protected alias를 추가 및 삭제할 수 있는 엑세스 권한을 부여할 사용자를 선택하세요.
{{< img src="/images/models/access_controls_admins.gif" alt="Managing registry admins" >}}


## Protected alias 추가하기
1. [W&B Model Registry 앱](https://wandb.ai/registry/model)으로 이동하세요.
2. 페이지 오른쪽 상단에 있는 톱니바퀴 버튼을 선택하세요.
{{< img src="/images/models/rbac_gear_button.png" alt="Registry settings gear button" >}}
3. **Protected Aliases** 섹션까지 스크롤하세요.
4. 플러스 아이콘(**+**)을 클릭하여 새로운 alias를 추가하세요.
{{< img src="/images/models/access_controls_add_protected_aliases.gif" alt="Adding protected aliases" >}}