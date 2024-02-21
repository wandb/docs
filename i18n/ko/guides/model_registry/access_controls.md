---
description: Use model registry role based access controls (RBAC) to control who can
  update protected aliases.
displayed_sidebar: default
---

# 데이터 거버넌스 및 엑세스 제어

*보호된 별칭*을 사용하여 모델 개발 파이프라인의 주요 단계를 나타내십시오. *모델 레지스트리 관리자*만 보호된 별칭을 추가, 수정 또는 제거할 수 있습니다. 모델 레지스트리 관리자는 보호된 별칭을 정의하고 사용할 수 있습니다. W&B는 비 관리자 사용자가 모델 버전에서 보호된 별칭을 추가하거나 제거하는 것을 차단합니다.

:::info
팀 관리자 또는 현재 레지스트리 관리자만 레지스트리 관리자 목록을 관리할 수 있습니다.
:::

예를 들어, `staging`과 `production`을 보호된 별칭으로 설정한다고 가정해 보겠습니다. 팀의 모든 구성원은 새 모델 버전을 추가할 수 있습니다. 그러나, 관리자만 `staging` 또는 `production` 별칭을 추가할 수 있습니다.

## 엑세스 제어 설정
다음 단계는 팀의 모델 레지스트리에 대한 엑세스 제어를 설정하는 방법을 설명합니다.

1. [https://wandb.ai/registry/model](https://wandb.ai/registry/model)에서 W&B 모델 레지스트리 앱으로 이동합니다.
2. 페이지 오른쪽 상단에 있는 톱니바퀴 버튼을 선택합니다.
![](/images/models/rbac_gear_button.png)
3. **레지스트리 관리자 관리** 버튼을 선택합니다.
4. **구성원** 탭에서 모델 버전에서 보호된 별칭을 추가 및 제거할 권한을 부여하려는 사용자를 선택합니다.
![](/images/models/access_controls_admins.gif)

## 보호된 별칭 추가
1. [https://wandb.ai/registry/model](https://wandb.ai/registry/model)에서 W&B 모델 레지스트리 앱으로 이동합니다.
2. 페이지 오른쪽 상단에 있는 톱니바퀴 버튼을 선택합니다.
![](/images/models/rbac_gear_button.png)
3. **보호된 별칭** 섹션으로 아래로 스크롤합니다.
4. 새 별칭을 추가하려면 플러스 아이콘(**+**)을 클릭합니다.
![](/images/models/access_controls_add_protected_aliases.gif)