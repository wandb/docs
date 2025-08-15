---
title: ID 및 엑세스 관리 (IAM)
cascade:
- url: guides/hosting/iam/:filename
menu:
  default:
    identifier: ko-guides-hosting-iam-_index
    parent: w-b-platform
url: guides/hosting/iam/org_team_struct
weight: 2
---

W&B 플랫폼에는 W&B 내에서 세 가지 IAM 스코프가 있습니다: [Organizations]({{< relref path="#organization" lang="ko" >}}), [Teams]({{< relref path="#team" lang="ko" >}}), 그리고 [Projects]({{< relref path="#project" lang="ko" >}}).

## Organization

*Organization*은 W&B 계정 또는 인스턴스의 최상위 스코프입니다. 계정이나 인스턴스에서 이루어지는 모든 작업은 이 최상위 스코프 내에서 진행됩니다. 여기에는 사용자 관리, 팀 관리, 팀 내 프로젝트 관리, 사용량 추적 등이 포함됩니다.

[Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}})를 사용하는 경우, 여러 개의 organization을 가질 수 있으며, 각각이 비즈니스 유닛, 개인 사용자, 다른 기업과의 공동 파트너십 등에 대응할 수 있습니다.

[Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}}) 또는 [Self-managed instance]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ko" >}})를 사용하는 경우에는 한 개의 organization에 해당됩니다. 회사에서 여러 개의 Dedicated Cloud 또는 Self-managed 인스턴스를 비즈니스 유닛이나 부서별로 나눠서 운영할 수도 있지만, 이는 AI 실무자 관리를 위한 선택사항입니다.

자세한 내용은 [조직 관리]({{< relref path="./access-management/manage-organization.md" lang="ko" >}})를 참고하세요.

## Team

*Team*은 organization 내의 하위 스코프로, 비즈니스 유닛/기능, 부서 또는 회사 내 프로젝트 팀과 매핑될 수 있습니다. 배포 유형과 요금제에 따라 한 organization에 여러 개의 team이 있을 수 있습니다.

AI 프로젝트는 team이라는 컨텍스트 내에서 조직됩니다. 팀 내 엑세스 제어는 팀 관리자가 담당하며, 팀 관리자는 반드시 상위 organization의 관리자일 필요는 없습니다.

자세한 내용은 [팀 추가 및 관리]({{< relref path="./access-management/manage-organization.md#add-and-manage-teams" lang="ko" >}})를 참고하세요.

## Project

*Project*는 team의 하위 스코프로, 구체적인 목표를 가진 실제 AI 프로젝트와 매핑됩니다. 한 team 안에 여러 개의 project가 있을 수 있습니다. 각 프로젝트는 접근 가능 대상을 결정하는 공개 범위(visibility mode)를 갖고 있습니다.

모든 프로젝트는 [Workspaces]({{< relref path="/guides/models/track/workspaces.md" lang="ko" >}})와 [Reports]({{< relref path="/guides/core/reports/" lang="ko" >}})로 구성되어 있으며, 관련 [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ko" >}}), [Sweeps]({{< relref path="/guides/models/sweeps/" lang="ko" >}}), [Automations]({{< relref path="/guides/core/automations/" lang="ko" >}})와 연결됩니다.