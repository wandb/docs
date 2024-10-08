---
title: W&B IAM structure
displayed_sidebar: default
---

W&B 플랫폼에는 W&B 내에 세 가지 IAM 범위가 있습니다: [Organizations](#organization), [Teams](#team), 그리고 [Projects](#project).

## Organization

*Organization*은 W&B 계정 또는 인스턴스의 루트 범위입니다. 계정 또는 인스턴스 내의 모든 작업은 사용자 관리, 팀 관리, 팀 내 프로젝트 관리, 사용 추적 등을 포함하여 이 루트 범위 내에서 이루어집니다.

[Multi-tenant Cloud](../hosting-options/saas_cloud.md)를 사용 중인 경우, 각 조직이 하나의 비즈니스 단위, 개인 사용자, 다른 비즈니스와의 공동 파트너십 등에 해당할 수 있는 여러 개의 조직이 있을 수 있습니다.

[전용 클라우드](../hosting-options/dedicated_cloud.md) 또는 [Self-managed instance](../hosting-options/self-managed.md)를 사용 중인 경우, 이는 하나의 조직에 해당합니다. 귀사의 경우, 여러 비즈니스 단위 또는 부서에 대응하기 위해 여러 개의 전용 클라우드나 Self-managed 인스턴스를 가질 수 있지만, 이는 철저히 비즈니스 또는 부서 전반에 걸쳐 AI 실무자를 관리하는 선택적인 방법입니다.

자세한 내용은 [Organizations](../../app/features/organizations.md)에서 확인하세요.

## Team

*Team*은 조직 내의 하위 범위로, 비즈니스 단위/기능, 부서 또는 회사의 프로젝트 팀에 대응할 수 있습니다. 배포 유형과 가격 계획에 따라 조직 내에 여러 팀이 있을 수 있습니다.

AI 프로젝트는 팀 내의 컨텍스트에서 조직됩니다. 팀 내 엑세스 제어는 팀 관리자에 의해 관리되며, 이들은 상위 조직 수준에서 관리자일 수도 있고 아닐 수도 있습니다.

자세한 내용은 [Teams](../../app/features/teams.md)에서 확인하세요.

## Project

*Project*는 팀 내의 하위 범위로, 특정한 의도된 결과를 가진 실제 AI 프로젝트에 대응합니다. 하나의 팀 내에 여러 프로젝트가 있을 수 있습니다. 각 프로젝트에는 엑세스할 수 있는 사용자를 결정하는 가시성 모드가 있습니다.

모든 프로젝트는 [Workspaces](../../app/pages/workspaces.md)와 [Reports](../../reports/intro.md)로 구성되며, 관련 [Artifacts](../../artifacts/intro.md), [Sweeps](../../sweeps/intro.md), [Launch Jobs](../../launch/intro.md), [Automations](../../artifacts/project-scoped-automations.md)와 연결되어 있습니다.