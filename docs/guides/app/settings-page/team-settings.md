---
title: Manage team settings
description: 팀 설정 페이지에서 팀 구성원, 아바타, 알림, 개인 정보 설정을 관리하세요.
displayed_sidebar: default
---

# 팀 설정

팀의 프로필 페이지로 이동하여 **Team settings** 아이콘을 선택하여 팀 설정을 관리하세요. 모든 팀 멤버가 팀 설정을 수정할 수 있는 것은 아닙니다. 팀 관리자는 팀의 설정을 보고 팀 수준의 TTL 설정에 엑세스할 수 있습니다. 멤버의 계정 유형(관리자, 멤버 또는 서비스)에 따라 해당 멤버가 수정할 수 있는 설정이 결정됩니다.

:::info
관리자 계정 유형만 팀 설정을 변경하거나 팀에서 멤버를 제거할 수 있습니다.
:::

## Members

**Members** 섹션에서는 가입 초대를 수락한 멤버와 대기 중인 초대 목록을 보여줍니다. 표시된 각 멤버는 이름, 사용자 이름 및 계정 유형이 표시됩니다. 계정 유형은 관리자(Admin), 멤버, 서비스 세 가지가 있습니다.

### 팀 내 멤버 역할 변경

팀 내 멤버 역할을 변경하려면 다음 단계를 완료하세요:

1. 특정 팀 멤버의 이름 옆에 있는 계정 유형 아이콘을 선택합니다. 모달이 나타납니다.
2. 드롭다운 메뉴를 선택합니다.
3. 드롭다운에서 해당 팀 멤버가 가지게 할 계정 유형을 선택합니다.

### 팀에서 멤버 제거

팀에서 제거할 멤버의 이름 옆에 있는 휴지통 아이콘을 선택하세요.

:::info
팀 계정에서 생성된 run은 해당 run을 생성한 멤버가 팀에서 제거되더라도 유지됩니다.
:::

### 가입 시 멤버를 팀 조직에 매칭

새로운 사용자가 가입할 때 조직 내 팀을 발견할 수 있도록 허용하세요. 새로운 사용자는 조직의 인증된 이메일 도메인과 일치하는 인증된 이메일 도메인을 가지고 있어야 합니다. 인증된 새로운 사용자는 W&B 계정에 가입할 때 조직에 속하는 인증된 팀 목록을 보게 됩니다.

조직의 관리자(Admin)가 이 기능을 활성화해야 합니다. 이 기능을 활성화하려면 다음 단계를 따르세요:

1. Teams Setting 페이지의 **Privacy** 섹션으로 이동합니다.
2. "Allow users with matching organization email domain to join this team"라는 문구 옆에 있는 **Claim Organization Email Domain** 버튼을 선택합니다.
3. 새롭게 활성화된 토글을 선택합니다.

## Avatar

**Avatar** 섹션으로 이동하여 이미지를 업로드하여 아바타를 설정하세요.

1. **Update Avatar**를 선택하여 파일 대화 상자가 나타나도록 합니다.
2. 파일 대화 상자에서 사용할 이미지를 선택합니다.

## Alerts

run이 충돌하거나 완료되거나 사용자 정의 알림을 설정할 때 팀에 알리세요. 팀은 이메일 또는 Slack을 통해 알림을 받을 수 있습니다.

받고자 하는 이벤트 유형 옆의 스위치를 토글하세요. Weights and Biases는 기본적으로 다음 이벤트 유형 옵션을 제공합니다:

* **Runs finished**: Weights and Biases run이 성공적으로 완료되었는지 여부.
* **Run crashed**: run이 완료에 실패했는지 여부.

알림 설정 및 관리에 대한 자세한 내용은 [Send alerts with wandb.alert](../../runs/alert.md)를 참조하세요.

## Privacy

**Privacy** 섹션으로 이동하여 개인 정보 설정을 변경하세요. 관리자 역할을 가진 멤버만 개인 정보 설정을 수정할 수 있습니다. 관리자는 다음을 할 수 있습니다:

* 팀의 프로젝트를 개인 프로젝트로 설정할 수 있습니다.
* 기본적으로 코드 저장을 활성화할 수 있습니다.

## Usage

**Usage** 섹션은 Weights and Biases 서버에서 팀이 소비한 총 메모리 사용량을 설명합니다. 기본 저장 계획은 100GB입니다. 저장 및 가격에 대한 자세한 내용은 [Pricing](https://wandb.ai/site/pricing) 페이지를 참조하세요.

## Storage

**Storage** 섹션은 팀의 데이터를 위해 사용 중인 클라우드 저장소 버킷 설정을 설명합니다. 자세한 내용은 [Secure Storage Connector](../features/teams.md#secure-storage-connector)를 참조하거나 자체 호스팅하는 경우 [W&B Server](../../hosting/data-security/secure-storage-connector.md) 문서를 확인하세요.