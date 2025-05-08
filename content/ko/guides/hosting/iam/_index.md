---
title: Identity and access management (IAM)
cascade:
- url: /ko/guides//hosting/iam/:filename
menu:
  default:
    identifier: ko-guides-hosting-iam-_index
    parent: w-b-platform
url: /ko/guides//hosting/iam/org_team_struct
weight: 2
---

W&B 플랫폼은 W&B 내에 세 가지 IAM 범위, 즉 [Organizations]({{< relref path="#organization" lang="ko" >}}), [Teams]({{< relref path="#team" lang="ko" >}}) 및 [Projects]({{< relref path="#project" lang="ko" >}})를 가지고 있습니다.

## Organization

*Organization* 은 W&B 계정 또는 인스턴스의 루트 범위입니다. 계정 또는 인스턴스의 모든 작업은 사용자 관리, 팀 관리, 팀 내 프로젝트 관리, 사용량 추적 등을 포함하여 해당 루트 범위 내에서 수행됩니다.

[Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}})를 사용하는 경우, 각 Organization은 사업부, 개인 사용자, 다른 기업과의 합작 파트너십 등에 해당할 수 있습니다.

[전용 클라우드(Dedicated Cloud)]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}}) 또는 [자체 관리 인스턴스]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ko" >}})를 사용하는 경우, 하나의 Organization에 해당합니다. 회사는 여러 사업부 또는 부서에 매핑하기 위해 여러 개의 전용 클라우드(Dedicated Cloud) 또는 자체 관리 인스턴스를 가질 수 있지만, 이는 사업 또는 부서 전체에서 AI 실무자를 관리하는 선택적인 방법입니다.

자세한 내용은 [Organizations 관리]({{< relref path="./access-management/manage-organization.md" lang="ko" >}})를 참조하십시오.

## Team

*Team* 은 Organization 내의 하위 범위로, 사업부/기능, 부서 또는 회사 내 프로젝트 팀에 매핑될 수 있습니다. 배포 유형 및 가격 계획에 따라 Organization 내에 여러 개의 Team이 있을 수 있습니다.

AI 프로젝트는 Team 컨텍스트 내에서 구성됩니다. Team 내의 엑세스 제어는 Team 관리자가 관리하며, 이들은 상위 Organization 수준의 관리자일 수도 아닐 수도 있습니다.

자세한 내용은 [Team 추가 및 관리]({{< relref path="./access-management/manage-organization.md#add-and-manage-teams" lang="ko" >}})를 참조하십시오.

## Project

*Project* 는 Team 내의 하위 범위로, 특정 의도된 결과를 가진 실제 AI 프로젝트에 매핑됩니다. Team 내에 여러 개의 Project가 있을 수 있습니다. 각 Project에는 누가 엑세스할 수 있는지 결정하는 가시성 모드가 있습니다.

모든 Project는 [Workspaces]({{< relref path="/guides/models/track/workspaces.md" lang="ko" >}}) 및 [Reports]({{< relref path="/guides/core/reports/" lang="ko" >}})로 구성되며, 관련 [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ko" >}}), [Sweeps]({{< relref path="/guides/models/sweeps/" lang="ko" >}}) 및 [Automations]({{< relref path="/guides/core/automations/" lang="ko" >}})에 연결됩니다.
