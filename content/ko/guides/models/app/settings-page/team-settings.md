---
title: 팀 설정 관리
description: Team Settings 페이지에서 팀 멤버, 아바타, 알림, 그리고 개인정보 설정을 관리할 수 있습니다.
menu:
  default:
    identifier: ko-guides-models-app-settings-page-team-settings
    parent: settings
weight: 30
---

# 팀 설정

팀 멤버, 아바타, 알림, 프라이버시, 사용량 등 팀의 설정을 변경할 수 있습니다. 조직 관리자와 팀 관리자는 팀 설정을 조회하고 수정할 수 있습니다.

{{% alert %}}
관리자 계정 유형만 팀 설정을 변경하거나 팀에서 멤버를 제거할 수 있습니다.
{{% /alert %}}

## 멤버
**Members** 섹션에서는 팀 초대가 대기 중이거나 이미 수락한 모든 멤버의 목록을 보여줍니다. 각 멤버에는 멤버의 이름, 사용자명, 이메일, 팀 역할, 그리고 Organization 으로부터 상속받은 Models 및 Weave 사용 권한이 표시됩니다. 표준 팀 역할은 **Admin**, **Member**, **View-only** 중에서 선택할 수 있습니다. 조직에서 [커스텀 역할]({{< relref path="/guides/hosting/iam/access-management/manage-organization.md#create-custom-roles" lang="ko" >}})을 생성했다면, 커스텀 역할도 지정할 수 있습니다.

팀 생성, 팀 관리, 팀 멤버십 및 역할 관리에 대한 자세한 내용은 [Add and Manage teams]({{< relref path="/guides/hosting/iam/access-management/manage-organization.md#add-and-manage-teams" lang="ko" >}})를 참고하세요. 새로운 멤버 초대 권한이나 기타 팀 프라이버시 설정은 [Privacy]({{< relref path="#privacy" lang="ko" >}})에서 변경할 수 있습니다.

## 아바타

**Avatar** 섹션으로 이동하여 이미지를 업로드하면 아바타를 설정할 수 있습니다.

1. **Update Avatar**를 선택하면 파일 선택 창이 열립니다.
2. 파일 선택 창에서 사용하고 싶은 이미지를 선택하세요.

## 알림

run 이 크래시나 완료될 때, 또는 커스텀 알림을 설정하여 팀에 알릴 수 있습니다. 팀은 이메일이나 Slack을 통해 알림을 받을 수 있습니다.

원하는 이벤트 타입 옆의 스위치를 켜서 알림 수신 여부를 설정하세요. Weights and Biases에서는 기본적으로 다음 이벤트 타입을 제공합니다:

* **Runs finished**: run 이 성공적으로 종료되었는지 여부
* **Run crashed**: run 이 비정상적으로 종료된 경우

알림 설정 및 관리 방법에 대한 자세한 내용은 [Send alerts with `wandb.Run.alert()`]({{< relref path="/guides/models/track/runs/alert.md" lang="ko" >}})를 참고하세요.

## Slack 알림
Registry나 프로젝트에서 새로운 artifact 생성 또는 run metric 임계값 도달 등 이벤트가 발생할 때, 팀의 [automations]({{< relref path="/guides/core/automations/" lang="ko" >}})이 Slack으로 알림을 보낼 수 있도록 Slack 목적지를 설정할 수 있습니다. 자세한 내용은 [Create a Slack automation]({{< relref path="/guides/core/automations/create-automations/slack.md" lang="ko" >}})을 참고하세요.

{{% pageinfo color="info" %}}
{{< readfile file="/_includes/enterprise-only.md" >}}
{{% /pageinfo %}}

## Webhook
Registry나 프로젝트에서 artifact 생성, run metric 조건 도달 등 이벤트 발생 시, 팀의 [automations]({{< relref path="/guides/core/automations/" lang="ko" >}})이 webhook을 실행할 수 있도록 설정할 수 있습니다. [Create a webhook automation]({{< relref path="/guides/core/automations/create-automations/slack.md" lang="ko" >}})을 참고하세요.

{{% pageinfo color="info" %}}
{{< readfile file="/_includes/enterprise-only.md" >}}
{{% /pageinfo %}}

## 프라이버시

**Privacy** 섹션에서 프라이버시 설정을 변경할 수 있습니다. 조직 관리자만 프라이버시 설정을 수정할 수 있습니다.

- 앞으로 생성되는 프로젝트를 공개하거나 리포트를 공개로 공유하는 기능을 비활성화할 수 있습니다.
- 팀 관리자뿐만 아니라 모든 팀 멤버가 다른 멤버를 초대할 수 있도록 허용할 수 있습니다.
- 코드 저장 기능의 기본 활성화 여부를 관리할 수 있습니다.

## 사용량

**Usage** 섹션에서는 팀이 Weights and Biases 서버에서 사용한 전체 메모리 사용량을 확인할 수 있습니다. 기본 저장 용량은 100GB입니다. 저장 공간 및 가격에 대한 더 자세한 정보는 [Pricing](https://wandb.ai/site/pricing) 페이지를 참고하세요.

## 스토리지

**Storage** 섹션에서는 팀의 데이터를 저장할 때 사용하는 클라우드 스토리지 버킷 설정 정보를 확인할 수 있습니다. 자세한 내용은 [Secure Storage Connector]({{< relref path="teams.md#secure-storage-connector" lang="ko" >}})를 참고하거나, 직접 호스팅하는 경우 [W&B Server]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ko" >}}) 문서를 확인하세요.