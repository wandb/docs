---
title: 설정에서 개인정보 보호 설정 구성
menu:
  default:
    identifier: ko-guides-hosting-privacy-settings
    parent: w-b-platform
weight: 4
---

조직과 팀 관리자는 각각 조직 및 팀 범위에서 개인정보 보호 설정을 구성할 수 있습니다. 조직 범위에서 설정된 경우, 조직 관리자는 해당 조직 내 모든 팀에 대해 해당 설정을 강제 적용합니다.

{{% alert %}}
W&B는 조직 관리자가 개인정보 보호 설정을 강제 적용하기 전에 해당 내용을 미리 모든 팀 관리자와 사용자에게 안내할 것을 권장합니다. 이는 워크플로우에 예상치 못한 변경이 발생하는 것을 방지하기 위함입니다.
{{% /alert %}}

## 팀 개인정보 보호 설정 구성하기

팀 관리자는 팀 **설정** 탭의 `Privacy` 섹션에서 개인정보 보호 설정을 직접 구성할 수 있습니다. 단, 조직 범위에서 강제 적용된 설정은 변경할 수 없습니다.

* 이 팀을 비멤버에게 숨기기
* 모든 신규 팀 Projects를 비공개로 설정 (공개 공유 불가)
* 모든 팀 멤버가 다른 멤버를 초대할 수 있도록 허용 (관리자만 초대할 수 있도록 제한하지 않음)
* 비공개 Projects의 Reports를 팀 외부에 공유하지 않도록 설정. 기존 매직 링크도 비활성화됩니다.
* 조직 이메일 도메인과 일치하는 사용자가 이 팀에 가입할 수 있도록 허용
    * 이 설정은 [SaaS Cloud]({{< relref path="./hosting-options/saas_cloud.md" lang="ko" >}})에서만 적용됩니다. [전용 클라우드]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ko" >}}) 또는 [자가 관리형]({{< relref path="/guides/hosting/hosting-options/self-managed/" lang="ko" >}}) 인스턴스에서는 지원되지 않습니다.
* 기본적으로 코드 저장 활성화

## 모든 팀에 개인정보 보호 설정 강제 적용하기

조직 관리자는 계정 또는 조직 대시보드의 **설정** 탭 내 `Privacy` 섹션에서 조직 내 모든 팀에 개인정보 보호 설정을 강제로 적용할 수 있습니다. 조직 관리자가 해당 설정을 강제 적용하면, 팀 관리자는 각 팀에서 해당 설정을 개별적으로 변경할 수 없습니다.

* 팀 노출 제한 강제 적용
    * 이 옵션을 활성화하면, 모든 팀이 비멤버에게 보이지 않도록 설정합니다.
* 향후 Projects의 비공개(또는 [제한됨]({{< relref path="./iam/access-management/restricted-projects.md" lang="ko" >}})) 상태 강제
    * 이 옵션을 활성화하면 모든 팀의 이후 생성되는 Projects가 자동으로 비공개 또는 제한됨으로 설정됩니다.
* 초대 권한 관리 강제 적용
    * 이 옵션을 활성화하면 관리자가 아닌 사용자는 어떤 팀에도 멤버 초대를 할 수 없습니다.
* Report 공유 제어 강제 적용
    * 이 옵션을 활성화하면 비공개 Projects의 Reports 외부 공유가 비활성화되며, 기존 매직 링크도 사용할 수 없습니다.
* 팀 자가 가입 제한 강제 적용
    * 이 옵션을 활성화하면 조직 이메일 도메인과 일치하는 사용자가 임의로 팀에 가입하는 것을 제한합니다.
    * 이 설정은 [SaaS Cloud]({{< relref path="./hosting-options/saas_cloud.md" lang="ko" >}})에서만 적용됩니다. [전용 클라우드]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ko" >}}) 또는 [자가 관리형]({{< relref path="/guides/hosting/hosting-options/self-managed/" lang="ko" >}}) 인스턴스에서는 지원되지 않습니다.
* 기본 코드 저장 비활성화 강제 적용
    * 이 옵션을 활성화하면 모든 팀에서 기본적으로 코드 저장이 비활성화됩니다.