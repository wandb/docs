---
title: Manage user settings
description: 사용자 설정에서 프로필 정보, 계정 기본 설정, 알림, 베타 제품 참여, GitHub 인테그레이션, 저장 공간 사용량, 계정
  활성화를 관리하고 팀을 만들 수 있습니다.
menu:
  default:
    identifier: ko-guides-models-app-settings-page-user-settings
    parent: settings
weight: 10
---

사용자 프로필 페이지로 이동하여 오른쪽 상단 모서리에 있는 사용자 아이콘을 선택하세요. 드롭다운 메뉴에서 **설정**을 선택합니다.

## 프로필

**프로필** 섹션에서는 계정 이름과 기관을 관리하고 수정할 수 있습니다. 선택적으로 자기소개, 위치, 개인 또는 기관 웹사이트 링크를 추가하고 프로필 이미지를 업로드할 수 있습니다.

## 팀

**팀** 섹션에서 새 팀을 만드세요. 새 팀을 만들려면 **새 팀** 버튼을 선택하고 다음을 제공하세요.

* **팀 이름** - 팀의 이름입니다. 팀 이름은 고유해야 합니다. 팀 이름은 변경할 수 없습니다.
* **팀 유형** - **업무** 또는 **학술** 버튼을 선택합니다.
* **회사/조직** - 팀의 회사 또는 조직 이름을 입력합니다. 드롭다운 메뉴에서 회사 또는 조직을 선택합니다. 선택적으로 새 조직을 제공할 수 있습니다.

{{% alert %}}
관리자 계정만 팀을 만들 수 있습니다.
{{% /alert %}}

## 베타 기능

**베타 기능** 섹션에서는 선택적으로 재미있는 애드온을 활성화하고 개발 중인 새 제품의 미리보기를 볼 수 있습니다. 활성화하려는 베타 기능 옆에 있는 토글 스위치를 선택합니다.

## 알림

[wandb.alert()]({{< relref path="/guides/models/track/runs/alert.md" lang="ko" >}})로 {Runs}이 충돌하거나 완료될 때 알림을 받고, 사용자 지정 알림을 설정하세요. 이메일이나 Slack을 통해 알림을 받으세요. 알림을 받을 이벤트 유형 옆에 있는 스위치를 토글하세요.

* **Runs finished**: Weights and Biases {Run}이 성공적으로 완료되었는지 여부입니다.
* **Run crashed**: {Run}이 완료되지 못한 경우 알림을 받습니다.

알림을 설정하고 관리하는 방법에 대한 자세한 내용은 [wandb.alert로 알림 보내기]({{< relref path="/guides/models/track/runs/alert.md" lang="ko" >}})를 참조하세요.

## 개인 GitHub {integration}

개인 Github 계정을 연결합니다. Github 계정을 연결하려면:

1. **Connect Github** 버튼을 선택합니다. 그러면 OAuth (Open Authorization) 페이지로 리디렉션됩니다.
2. **Organization access** 섹션에서 엑세스 권한을 부여할 조직을 선택합니다.
3. **Authorize** **wandb**를 선택합니다.

## 계정 삭제

**Delete Account** 버튼을 선택하여 계정을 삭제합니다.

{{% alert color="secondary" %}}
계정 삭제는 되돌릴 수 없습니다.
{{% /alert %}}

## 스토리지

**Storage** 섹션은 계정에서 Weights and Biases {서버}에서 사용한 총 메모리 사용량을 설명합니다. 기본 스토리지 요금제는 100GB입니다. 스토리지 및 가격 책정에 대한 자세한 내용은 [Pricing](https://wandb.ai/site/pricing) 페이지를 참조하세요.
