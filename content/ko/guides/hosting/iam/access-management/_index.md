---
title: 엑세스 관리
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

조직 도메인으로 W&B에 처음 가입한 사용자는 해당 조직의 *인스턴스 관리자 역할*로 지정됩니다. 조직 관리자는 특정 사용자에게 팀 관리자 역할을 할당합니다.

{{% alert %}}
조직 내에 인스턴스 관리자를 두 명 이상 두는 것을 W&B에서 권장합니다. 주된 관리자가 부재 시에도 관리 작업이 지속될 수 있도록 하는 것이 모범 사례입니다.
{{% /alert %}}

*팀 관리자*는 조직 내에서 팀 단위로 관리자 권한을 가지는 사용자입니다.

조직 관리자는 `https://wandb.ai/account-settings/`에서 조직의 계정 설정에 엑세스하여 사용자를 초대하고, 사용자의 역할을 할당하거나 수정하며, 팀 생성, 조직에서 사용자 삭제, 결제 관리자 지정 등 다양한 작업을 할 수 있습니다. 자세한 내용은 [사용자 추가 및 관리]({{< relref path="./manage-organization.md#add-and-manage-users" lang="ko" >}})를 참고하세요.

조직 관리자가 팀을 생성한 후에는 인스턴스 관리자나 팀 관리자가 다음 작업을 할 수 있습니다.

- 기본적으로, 오직 관리자만 팀에 사용자를 초대할 수 있고 팀에서 사용자를 삭제할 수 있습니다. 이 행동을 변경하려면 [팀 설정]({{< relref path="/guides/models/app/settings-page/team-settings.md#privacy" lang="ko" >}})을 참고하세요.
- 팀 멤버의 역할을 할당하거나 수정할 수 있습니다.
- 신규 사용자가 조직에 합류할 때, 자동으로 특정 팀에 추가될 수 있습니다.

조직 관리자와 팀 관리자는 둘 다 `https://wandb.ai/<your-team-name>`에 있는 팀 대시보드를 사용하여 팀을 관리합니다. 팀의 기본 개인정보 보호 설정 구성 등 자세한 내용은 [팀 추가 및 관리]({{< relref path="./manage-organization.md#add-and-manage-teams" lang="ko" >}})를 참고하세요.

## 관리자 엑세스 유지하기

항상 인스턴스 또는 조직 내에 최소한 한 명의 관리자 사용자가 존재해야 합니다. 그렇지 않으면, 그 누구도 조직의 W&B 계정을 설정하거나 유지할 수 없습니다.

사용자를 대화식으로 관리하는 경우, 다른 관리자 사용자를 삭제할 때도 관리자 엑세스가 필요합니다. 이는 단독 관리자 사용자가 실수로 삭제되는 위험을 줄여줍니다.

하지만 조직에서 자동화된 프로세스를 통해 W&B에서 사용자를 해지하는 경우, 해지 작업 중 마지막으로 남은 관리자가 인스턴스나 조직에서 의도치 않게 삭제될 수 있습니다.

운영 절차 개발이나 관리자 권한 복구가 필요하다면 [support](mailto:support@wandb.com)로 문의하세요.

## 특정 프로젝트에 대한 가시성 제한하기

W&B 프로젝트의 범위를 정의하여 누가 해당 프로젝트를 보고, 수정하고, W&B run을 제출할 수 있는지 제한할 수 있습니다. 프로젝트의 열람 대상을 제한하면, 팀이 민감하거나 기밀성이 있는 데이터를 다룰 때 특히 유용합니다.

조직 관리자, 팀 관리자, 또는 프로젝트 소유자는 프로젝트의 가시성을 직접 설정하고 수정할 수 있습니다.

더 자세한 내용은 [Project visibility]({{< relref path="./restricted-projects.md" lang="ko" >}})를 참고하세요.