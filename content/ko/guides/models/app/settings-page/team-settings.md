---
title: Manage team settings
description: '**Team Settings** 페이지에서 팀의 멤버, 아바타, 알림, 개인 정보 설정을 관리하세요.'
menu:
  default:
    identifier: ko-guides-models-app-settings-page-team-settings
    parent: settings
weight: 30
---

# 팀 설정

팀원, 아바타, 알림, 개인 정보 보호 및 사용량을 포함한 팀 설정을 변경합니다. 팀 관리자만 팀 설정을 보고 편집할 수 있습니다.

{{% alert %}}
관리 계정 유형만 팀 설정을 변경하거나 팀에서 팀원을 제거할 수 있습니다.
{{% /alert %}}

## 팀원
팀원 섹션에는 보류 중인 초대와 팀 가입 초대를 수락한 팀원 목록이 표시됩니다. 나열된 각 팀원에게는 팀원의 이름, 사용자 이름, 이메일, 팀 역할과 Organizations에서 상속받은 Models 및 Weave에 대한 엑세스 권한이 표시됩니다. 관리자(Admin), 팀원 및 보기 전용의 세 가지 표준 팀 역할이 있습니다.

팀 생성, 팀에 Users 초대, 팀에서 Users 제거 및 User의 역할 변경 방법에 대한 자세한 내용은 [팀 추가 및 관리]({{< relref path="/guides/hosting/iam/access-management/manage-organization.md#add-and-manage-teams" lang="ko" >}})를 참조하세요.

## 아바타

**아바타** 섹션으로 이동하여 이미지를 업로드하여 아바타를 설정합니다.

1. **아바타 업데이트**를 선택하여 파일 대화 상자를 표시합니다.
2. 파일 대화 상자에서 사용할 이미지를 선택합니다.

## 알림

Runs이 충돌하거나 완료될 때 또는 사용자 지정 알림을 설정할 때 팀에 알립니다. 팀은 이메일 또는 Slack을 통해 알림을 받을 수 있습니다.

알림을 받을 이벤트 유형 옆에 있는 스위치를 토글합니다. Weights & Biases는 기본적으로 다음과 같은 이벤트 유형 옵션을 제공합니다.

* **Runs 완료**: Weights & Biases run이 성공적으로 완료되었는지 여부입니다.
* **Run 충돌**: run이 완료되지 못한 경우입니다.

알림을 설정하고 관리하는 방법에 대한 자세한 내용은 [wandb.alert로 알림 보내기]({{< relref path="/guides/models/track/runs/alert.md" lang="ko" >}})를 참조하세요.

## 개인 정보 보호

**개인 정보 보호** 섹션으로 이동하여 개인 정보 보호 설정을 변경합니다. 관리자 역할이 있는 팀원만 개인 정보 보호 설정을 수정할 수 있습니다. 관리자 역할은 다음을 수행할 수 있습니다.

* 팀의 Projects을 비공개로 강제 설정합니다.
* 기본적으로 코드 저장을 활성화합니다.

## 사용량

**사용량** 섹션에서는 팀이 Weights & Biases 서버에서 소비한 총 메모리 사용량을 설명합니다. 기본 스토리지 플랜은 100GB입니다. 스토리지 및 가격 책정에 대한 자세한 내용은 [가격 책정](https://wandb.ai/site/pricing) 페이지를 참조하세요.

## 스토리지

**스토리지** 섹션에서는 팀의 data에 사용되는 클라우드 스토리지 버킷 설정을 설명합니다. 자세한 내용은 [보안 스토리지 커넥터]({{< relref path="teams.md#secure-storage-connector" lang="ko" >}})를 참조하거나 자체 호스팅하는 경우 [W&B 서버]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ko" >}}) 문서를 확인하세요.
