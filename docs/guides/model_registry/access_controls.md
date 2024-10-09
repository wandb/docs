---
title: Manage data governance and access control
description: 모델 레지스트리 역할 기반 엑세스 제어(RBAC)를 사용하여 보호된 에일리어스를 업데이트할 수 있는 사람을 제어하세요.
displayed_sidebar: default
---

*보호된 에일리어스*를 사용하여 모델 개발 파이프라인의 주요 단계를 나타내세요. *모델 레지스트리 관리자*만 보호된 에일리어스를 추가, 수정 또는 제거할 수 있습니다. 모델 레지스트리 관리자는 보호된 에일리어스를 정의하고 사용할 수 있습니다. W&B는 관리자 이외의 사용자가 모델 버전에서 보호된 에일리어스를 추가하거나 제거하지 못하도록 차단합니다.

:::info
팀 관리자 또는 현재 레지스트리 관리자만 레지스트리 관리자의 목록을 관리할 수 있습니다.
:::

예를 들어, `staging`과 `production`을 보호된 에일리어스로 설정했다고 가정해 보세요. 팀의 모든 멤버는 새로운 모델 버전을 추가할 수 있습니다. 하지만, 관리자만이 `staging` 또는 `production` 에일리어스를 추가할 수 있습니다.

## 엑세스 제어 설정
다음 단계는 팀의 모델 레지스트리에 대한 엑세스 제어를 설정하는 방법을 설명합니다.

1. W&B Model Registry 앱으로 이동합니다: [https://wandb.ai/registry/model](https://wandb.ai/registry/model).
2. 페이지 오른쪽 상단의 톱니 바퀴 버튼을 선택합니다.
![](/images/models/rbac_gear_button.png)
3. **레지스트리 관리자 관리** 버튼을 선택합니다. 
4. **Members** 탭에서, 모델 버전에서 보호된 에일리어스를 추가 및 제거할 수 있는 엑세스를 부여하려는 사용자를 선택합니다.
![](/images/models/access_controls_admins.gif)

## 보호된 에일리어스 추가
1. W&B Model Registry 앱으로 이동합니다: [https://wandb.ai/registry/model](https://wandb.ai/registry/model).
2. 페이지 오른쪽 상단의 톱니 바퀴 버튼을 선택합니다.
![](/images/models/rbac_gear_button.png)
3. **Protected Aliases** 섹션으로 스크롤합니다.
4. 새로운 에일리어스를 추가하려면 플러스 아이콘 (**+**)을 클릭합니다.
![](/images/models/access_controls_add_protected_aliases.gif)