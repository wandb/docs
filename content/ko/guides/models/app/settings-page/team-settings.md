---
title: Manage team settings
description: Team 설정 페이지에서 팀의 멤버, 아바타, 알림 및 개인 정보 설정을 관리하세요.
menu:
  default:
    identifier: ko-guides-models-app-settings-page-team-settings
    parent: settings
weight: 30
---

# 팀 설정

팀 멤버, 아바타, 알림, 개인 정보 보호, 사용량 등 팀 설정을 변경합니다. 조직 관리자와 팀 관리자는 팀 설정을 보고 편집할 수 있습니다.

{{% alert %}}
관리 계정 유형만 팀 설정을 변경하거나 팀에서 멤버를 제거할 수 있습니다.
{{% /alert %}}

## 멤버
멤버 섹션에는 보류 중인 초대 목록과 팀 가입 초대를 수락한 멤버가 모두 표시됩니다. 나열된 각 멤버는 멤버의 이름, 사용자 이름, 이메일, 팀 역할과 조직에서 상속된 Models 및 Weave에 대한 엑세스 권한을 표시합니다. 표준 팀 역할인 **Admin**, **Member** 및 **View-only** 중에서 선택할 수 있습니다. 조직에서 [사용자 정의 역할]({{< relref path="/guides/hosting/iam/access-management/manage-organization.md#create-custom-roles" lang="ko" >}})을 생성한 경우 사용자 정의 역할을 대신 할당할 수 있습니다.

팀 생성, 팀 관리, 팀 멤버십 및 역할 관리에 대한 자세한 내용은 [팀 추가 및 관리]({{< relref path="/guides/hosting/iam/access-management/manage-organization.md#add-and-manage-teams" lang="ko" >}})를 참조하세요. 팀에 대한 새로운 멤버 초대 권한을 구성하고 기타 개인 정보 보호 설정을 구성하려면 [개인 정보 보호]({{< relref path="#privacy" lang="ko" >}})를 참조하세요.

## 아바타

**아바타** 섹션으로 이동하여 이미지를 업로드하여 아바타를 설정합니다.

1. **아바타 업데이트**를 선택하여 파일 대화 상자를 표시합니다.
2. 파일 대화 상자에서 사용할 이미지를 선택합니다.

## 알림

Runs이 충돌하거나 완료될 때 또는 사용자 정의 알림을 설정할 때 팀에 알립니다. 팀은 이메일 또는 Slack을 통해 알림을 받을 수 있습니다.

알림을 받을 이벤트 유형 옆에 있는 스위치를 토글합니다. Weights & Biases는 기본적으로 다음과 같은 이벤트 유형 옵션을 제공합니다.

* **Runs finished**: Weights & Biases run이 성공적으로 완료되었는지 여부.
* **Run crashed**: run이 완료되지 못한 경우.

알림을 설정하고 관리하는 방법에 대한 자세한 내용은 [wandb.alert로 알림 보내기]({{< relref path="/guides/models/track/runs/alert.md" lang="ko" >}})를 참조하세요.

## Slack 알림
새로운 아티팩트가 생성되거나 run 메트릭이 정의된 임계값을 충족하는 경우와 같이 팀의 [자동화]({{< relref path="/guides/core/automations/" lang="ko" >}})가 Registry 또는 프로젝트에서 이벤트가 발생할 때 알림을 보낼 수 있는 Slack 대상을 구성합니다. [Slack 자동화 생성]({{< relref path="/guides/core/automations/create-automations/slack.md" lang="ko" >}})를 참조하세요.

{{% pageinfo color="info" %}}
{{< readfile file="/_includes/enterprise-only.md" >}}
{{% /pageinfo %}}

## 웹훅
새로운 아티팩트가 생성되거나 run 메트릭이 정의된 임계값을 충족하는 경우와 같이 팀의 [자동화]({{< relref path="/guides/core/automations/" lang="ko" >}})가 Registry 또는 프로젝트에서 이벤트가 발생할 때 실행할 수 있는 웹훅을 구성합니다. [웹훅 자동화 생성]({{< relref path="/guides/core/automations/create-automations/slack.md" lang="ko" >}})를 참조하세요.

{{% pageinfo color="info" %}}
{{< readfile file="/_includes/enterprise-only.md" >}}
{{% /pageinfo %}}

## 개인 정보 보호

**개인 정보 보호** 섹션으로 이동하여 개인 정보 보호 설정을 변경합니다. 조직 관리자만 개인 정보 보호 설정을 수정할 수 있습니다.

- 향후 프로젝트를 공개하거나 Reports를 공개적으로 공유하는 기능을 끕니다.
- 팀 관리자뿐만 아니라 모든 팀 멤버가 다른 멤버를 초대할 수 있도록 허용합니다.
- 코드 저장을 기본적으로 켤지 여부를 관리합니다.

## 사용량

**사용량** 섹션에서는 팀이 Weights and Biases 서버에서 소비한 총 메모리 사용량을 설명합니다. 기본 스토리지 플랜은 100GB입니다. 스토리지 및 가격 책정에 대한 자세한 내용은 [가격 책정](https://wandb.ai/site/pricing) 페이지를 참조하세요.

## 스토리지

**스토리지** 섹션에서는 팀의 데이터에 사용되는 클라우드 스토리지 버킷 구성을 설명합니다. 자세한 내용은 [보안 스토리지 커넥터]({{< relref path="teams.md#secure-storage-connector" lang="ko" >}})를 참조하거나 자체 호스팅하는 경우 [W&B 서버]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ko" >}}) 문서를 확인하세요.
