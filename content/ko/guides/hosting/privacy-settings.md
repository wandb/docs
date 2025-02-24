---
title: Configure privacy settings
menu:
  default:
    identifier: ko-guides-hosting-privacy-settings
    parent: w-b-platform
weight: 4
---

Organization 및 Team 관리자는 organization 및 team 범위에서 각각 일련의 개인 정보 보호 설정을 구성할 수 있습니다. organization 범위에서 구성된 경우 organization 관리자는 해당 organization의 모든 Teams에 대해 이러한 설정을 적용합니다.

{{% alert %}}
W&B는 organization 관리자가 organization 내의 모든 Team 관리자 및 사용자에게 미리 알린 후에만 개인 정보 보호 설정을 적용할 것을 권장합니다. 이는 워크플로우에서 예기치 않은 변경을 방지하기 위함입니다.
{{% /alert %}}

## Team의 개인 정보 보호 설정 구성

Team 관리자는 Team **Settings** 탭의 `Privacy` 섹션에서 각 Team에 대한 개인 정보 보호 설정을 구성할 수 있습니다. 각 설정은 organization 범위에서 적용되지 않는 한 구성할 수 있습니다.

* 이 Team을 모든 비 멤버에게 숨기기
* 향후 모든 Team Projects를 비공개로 만들기 (공개 공유 불가)
* 모든 Team 멤버가 다른 멤버를 초대하도록 허용 (관리자만 해당되지 않음)
* 비공개 Projects의 Reports에 대한 Team 외부로의 공개 공유를 해제합니다. 이렇게 하면 기존의 매직 링크가 해제됩니다.
* organization 이메일 도메인이 일치하는 사용자가 이 Team에 가입하도록 허용합니다.
    * 이 설정은 [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}})에만 적용됩니다. [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}}) 또는 [Self-managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ko" >}}) 인스턴스에서는 사용할 수 없습니다.
* 코드 저장을 기본적으로 활성화합니다.

## 모든 Teams에 대해 개인 정보 보호 설정 적용

Organization 관리자는 계정 또는 organization 대시보드의 **Settings** 탭의 `Privacy` 섹션에서 organization의 모든 Teams에 대한 개인 정보 보호 설정을 적용할 수 있습니다. Organization 관리자가 설정을 적용하면 Team 관리자는 각 Team 내에서 해당 설정을 구성할 수 없습니다.

* Team 가시성 제한 적용
    * 모든 Teams를 비 멤버에게 숨기려면 이 옵션을 활성화하십시오.
* 향후 Projects에 대한 개인 정보 보호 적용
    * 모든 Teams의 향후 모든 Projects를 비공개 또는 [restricted]({{< relref path="/guides/hosting/iam/access-management/restricted-projects.md" lang="ko" >}})로 적용하려면 이 옵션을 활성화하십시오.
* 초대 제어 적용
    * 관리자가 아닌 사람이 Team에 멤버를 초대하지 못하도록 하려면 이 옵션을 활성화하십시오.
* Report 공유 제어 적용
    * 비공개 Projects에서 Reports의 공개 공유를 해제하고 기존 매직 링크를 비활성화하려면 이 옵션을 활성화하십시오.
* Team 자체 가입 제한 적용
    * organization 이메일 도메인이 일치하는 사용자가 Team에 자체 가입하는 것을 제한하려면 이 옵션을 활성화하십시오.
    * 이 설정은 [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}})에만 적용됩니다. [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}}) 또는 [Self-managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ko" >}}) 인스턴스에서는 사용할 수 없습니다.
* 기본 코드 저장 제한 적용
    * 모든 Teams에 대해 기본적으로 코드 저장을 끄려면 이 옵션을 활성화하십시오.