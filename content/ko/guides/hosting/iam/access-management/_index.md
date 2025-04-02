---
title: Access management
cascade:
- url: guides/hosting/iam/access-management/:filename
menu:
  default:
    identifier: ko-guides-hosting-iam-access-management-_index
    parent: identity-and-access-management-iam
url: guides/hosting/iam/access-management-intro
weight: 2
---

## 조직 내에서 사용자 및 팀 관리하기
고유한 조직 도메인으로 W&B에 처음 가입하는 사용자는 해당 조직의 *인스턴스 관리자 역할*로 지정됩니다. 조직 관리자는 특정 사용자에게 팀 관리자 역할을 할당합니다.

{{% alert %}}
W&B는 조직에 둘 이상의 인스턴스 관리자를 두는 것을 권장합니다. 이는 기본 관리자가 없을 때 관리 작업을 계속할 수 있도록 보장하는 가장 좋은 방법입니다.
{{% /alert %}}

*팀 관리자*는 팀 내에서 관리 권한을 가진 조직의 사용자입니다.

조직 관리자는 `https://wandb.ai/account-settings/`에서 조직의 계정 설정을 엑세스하고 사용하여 사용자를 초대하고, 사용자의 역할을 할당하거나 업데이트하고, 팀을 만들고, 조직에서 사용자를 제거하고, 청구 관리자를 할당하는 등의 작업을 수행할 수 있습니다. 자세한 내용은 [사용자 추가 및 관리]({{< relref path="./manage-organization.md#add-and-manage-users" lang="ko" >}})를 참조하세요.

조직 관리자가 팀을 생성하면 인스턴스 관리자 또는 팀 관리자는 다음을 수행할 수 있습니다.

- 기본적으로 관리자만 해당 팀에 사용자를 초대하거나 팀에서 사용자를 제거할 수 있습니다. 이 행동 을 변경하려면 [팀 설정]({{< relref path="/guides/models/app/settings-page/team-settings.md#privacy" lang="ko" >}})을 참조하세요.
- 팀 멤버의 역할을 할당하거나 업데이트합니다.
- 새 사용자가 조직에 가입할 때 자동으로 팀에 추가합니다.

조직 관리자와 팀 관리자는 모두 `https://wandb.ai/<your-team-name>`에서 팀 대시보드 를 사용하여 팀을 관리합니다. 자세한 내용과 팀의 기본 개인 정보 보호 설정을 구성하려면 [팀 추가 및 관리]({{< relref path="./manage-organization.md#add-and-manage-teams" lang="ko" >}})를 참조하세요.

## 특정 Projects 에 대한 가시성 제한

W&B project 의 범위를 정의하여 누가 W&B runs 를 보고, 편집하고, 제출할 수 있는지 제한합니다. 팀이 민감하거나 기밀 데이터를 다루는 경우 project 를 볼 수 있는 사람을 제한하는 것이 특히 유용합니다.

조직 관리자, 팀 관리자 또는 project 소유자는 project 의 가시성을 설정하고 편집할 수 있습니다.

자세한 내용은 [Project visibility]({{< relref path="./restricted-projects.md" lang="ko" >}})를 참조하세요.
