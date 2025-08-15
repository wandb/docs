---
title: 릴리스 정책 및 프로세스
description: W&B Server 배포 프로세스
date: 2025-05-01
menu:
  default:
    identifier: server-release-process
    parent: w-b-platform
  reference:
    identifier: ko-ref-release-notes-release-policies
weight: 20
---

이 페이지는 W&B Server 릴리스 및 W&B의 릴리스 정책에 대한 세부 정보를 제공합니다. 이 페이지는 [W&B 전용 클라우드]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ko" >}}) 및 [셀프 관리형]({{< relref path="/guides/hosting/hosting-options/self-managed/" lang="ko" >}}) 배포에 관련된 내용입니다. 개별 W&B Server 릴리스에 대해 더 알고 싶으시다면 [W&B 릴리스 노트]({{< relref path="/ref/release-notes/" lang="ko" >}})를 참고하세요.

W&B가 [W&B 다중 테넌트 클라우드]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}})는 완전히 관리하므로, 이 페이지의 세부 내용은 해당 환경에 적용되지 않습니다.

## 릴리스 지원 및 EOL(지원 종료) 정책
W&B는 각 주요 W&B Server 릴리스를 최초 릴리스일부터 12개월간 지원합니다.
- **전용 클라우드** 인스턴스는 자동으로 업데이트되어 지원이 유지됩니다.
- **셀프 관리형** 인스턴스를 사용하는 고객은 직접 업그레이드를 통해 지원이 유지되도록 해야 합니다. 지원이 종료된 버전에 머무르는 것은 피해주세요.

  {{% alert %}}
  W&B는 **셀프 관리형** 인스턴스를 사용하는 고객이 최소 분기마다 한 번 최신 릴리스로 배포를 업데이트하여 지원을 받고 최신 기능, 성능 개선 및 수정 사항을 받을 것을 강력히 권장합니다.
  {{% /alert %}}

## 릴리스 유형 및 주기
- **주요 릴리스**는 매달 배포되며, 신규 기능, 기능 개선, 성능 향상, 중·저 위험도의 버그 수정, 기능 중단 등이 포함될 수 있습니다. 예시: `0.68.0`
- **패치 릴리스**는 주요 버전 내에서 필요에 따라 배포되며, 치명적이거나 심각한 버그 수정이 포함됩니다. 예시: `0.67.1`

## 릴리스 배포 과정
1. 테스트 및 검증이 완료되면, 우선 모든 **전용 클라우드** 인스턴스에 릴리스를 적용하여 완전히 최신 상태를 유지합니다.
1. 추가 관찰 이후, 릴리스가 공개되어 **셀프 관리형** 배포 환경에서는 각자의 일정에 맞춰 업그레이드할 수 있으며, [릴리스 지원 및 지원 종료(EOL) 정책]({{< relref path="#release-support-and-end-of-life-policy" lang="ko" >}})을 준수하여 적시에 업그레이드해야 합니다. [W&B Server 업그레이드 방법]({{< relref path="/guides/hosting/hosting-options/self-managed/server-upgrade-process.md" lang="ko" >}})에 대해 자세히 알아보세요.

## 업그레이드 시 다운타임
- **전용 클라우드** 인스턴스를 업그레이드할 때 일반적으로 다운타임은 발생하지 않으나, 다음과 같은 상황에서는 예외가 있을 수 있습니다:
  - 신규 기능 또는 기능 개선을 위해 인프라(컴퓨트, 스토리지, 네트워크 등)의 변경이 필요한 경우
  - 보안 수정 등 중요한 인프라 변경 사항을 적용하는 경우
  - 인스턴스의 현재 버전이 [지원 종료(EOL)]({{< relref path="/guides/hosting/hosting-options/self-managed/server-upgrade-process.md" lang="ko" >}}) 상태에 도달하여 지원 유지를 위해 W&B가 업그레이드하는 경우
- **셀프 관리형** 배포에서는, 고객이 서비스 수준 목표(SLO)를 만족시키도록 롤링 업데이트 프로세스를 직접 구현해야 합니다(예: [Kubernetes에서 W&B Server 운영]({{< relref path="/guides/hosting/hosting-options/self-managed/kubernetes-operator/" lang="ko" >}})).

## 기능 사용 가능 시점
설치 또는 업그레이드 후 일부 기능은 즉시 사용이 불가능할 수 있습니다.

### 엔터프라이즈 기능
엔터프라이즈 라이선스는 중요한 보안 및 엔터프라이즈 환경에 적합한 추가 기능 지원을 포함합니다. 일부 고급 기능은 엔터프라이즈 라이선스가 필요합니다.

- **전용 클라우드** 인스턴스에는 엔터프라이즈 라이선스가 포함되어 있으므로 별도 조치가 필요 없습니다.
- **셀프 관리형** 배포에서는, 엔터프라이즈 라이선스가 설정되기 전까지 해당 기능을 사용할 수 없습니다. 자세한 내용이나 라이선스 획득 방법은 [W&B Server 라이선스 획득하기]({{< relref path="/guides/hosting/hosting-options/self-managed.md#obtain-your-wb-server-license" lang="ko" >}})를 참고하세요.

### 프라이빗 프리뷰 및 옵트인 기능
대부분의 기능은 W&B Server 설치 또는 업그레이드 직후 바로 사용할 수 있습니다. 일부 기능은 W&B 팀에서 활성화해야 사용할 수 있습니다.

{{% alert color="warning" %}}
프리뷰 단계의 기능은 언제든지 변경될 수 있습니다. 프리뷰 기능이 일반에 공개될 것이라는 보장은 없습니다.
{{% /alert %}}

- **프라이빗 프리뷰**: W&B가 디자인 파트너와 초기 도입자를 초대하여 해당 기능을 테스트하고 피드백을 받습니다. 프라이빗 프리뷰 기능은 프로덕션 환경에서 사용하는 것을 권장하지 않습니다.

    해당 기능을 사용하려면 W&B 팀이 인스턴스에 기능을 활성화해야 하며, 공개 문서는 제공되지 않고 지침은 개별적으로 전달됩니다. 인터페이스와 API가 변경될 수 있고, 기능이 완전히 구현되지 않았을 수 있습니다.
- **퍼블릭 프리뷰**: 정식 공개 전에 사용해보고 싶다면 W&B에 연락하여 퍼블릭 프리뷰에 옵트인할 수 있습니다.

    퍼블릭 프리뷰 기능도 W&B 팀이 직접 활성화해야 하며, 문서가 완전하지 않을 수 있고 인터페이스와 API가 변경되거나 기능이 완전히 구현되지 않았을 수 있습니다.

각 W&B Server 릴리스 및 기능 제한 사항에 대해 더 알고 싶으시다면 [W&B 릴리스 노트]({{< relref path="/ref/release-notes/" lang="ko" >}})를 참고하세요.