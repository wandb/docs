---
title: 전용 클라우드에 대한 프라이빗 연결 구성
menu:
  default:
    identifier: ko-guides-hosting-data-security-private-connectivity
    parent: data-security
weight: 4
---

[전용 클라우드]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ko" >}}) 인스턴스에 클라우드 제공자의 안전한 프라이빗 네트워크를 통해 연결할 수 있습니다. 이는 AI 워크로드에서 W&B API로 엑세스하거나, 선택적으로 사용자 브라우저에서 W&B 앱 UI로 엑세스할 때도 적용됩니다. 프라이빗 연결을 사용할 경우, 관련 요청 및 응답은 공용 네트워크나 인터넷을 통하지 않습니다.

{{% alert %}}
전용 클라우드에서 고급 보안 옵션으로 안전한 프라이빗 연결 기능이 곧 제공될 예정입니다.
{{% /alert %}}

안전한 프라이빗 연결은 AWS, GCP, Azure에서 전용 클라우드 인스턴스에서 사용할 수 있습니다:

* AWS에서는 [AWS Privatelink](https://aws.amazon.com/privatelink/) 사용
* GCP에서는 [GCP Private Service Connect](https://cloud.google.com/vpc/docs/private-service-connect) 사용
* Azure에서는 [Azure Private Link](https://azure.microsoft.com/products/private-link) 사용

기능이 활성화되면, W&B가 인스턴스를 위한 프라이빗 엔드포인트 서비스를 생성하고 연결에 필요한 DNS URI를 제공합니다. 이를 통해, 클라우드 계정 내에서 프라이빗 엔드포인트를 생성하여 관련 트래픽을 프라이빗 엔드포인트 서비스로 라우팅할 수 있습니다. 프라이빗 엔드포인트는 클라우드 VPC 또는 VNet 내에서 실행 중인 AI 트레이닝 워크로드에 대해 더 쉽게 설정할 수 있습니다. 사용자 브라우저에서 W&B 앱 UI로의 트래픽에 동일한 방식을 사용하기 위해서는, 사내 네트워크에서 클라우드 계정 내 프라이빗 엔드포인트로의 DNS 기반 라우팅을 적절히 구성해야 합니다.

{{% alert %}}
이 기능을 사용하고 싶으신 경우, W&B 팀에 문의해 주세요.
{{% /alert %}}

[IP 허용 리스트]({{< relref path="./ip-allowlisting.md" lang="ko" >}})와 함께 안전한 프라이빗 연결을 사용할 수 있습니다. IP 허용 리스트용으로 안전한 프라이빗 연결을 사용할 경우, W&B에서는 AI 워크로드에서 나가는 모든 트래픽과 가능한 다수의 사용자 브라우저 트래픽을 모두 프라이빗 연결로 보호하는 것을 권장하며, 특권 위치에서는 인스턴스 관리용으로만 IP 허용 리스트를 사용하는 것이 좋습니다.