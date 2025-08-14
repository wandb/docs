---
title: Configure private connectivity to Dedicated Cloud
menu:
  default:
    identifier: ko-guides-hosting-data-security-private-connectivity
    parent: data-security
weight: 4
---

클라우드 공급자의 보안 사설 네트워크를 통해 [전용 클라우드]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ko" >}}) 인스턴스에 연결할 수 있습니다. 이는 AI 워크로드에서 W&B API로의 엑세스와 선택적으로 사용자 브라우저에서 W&B 앱 UI로의 엑세스에도 적용됩니다. 사설 연결을 사용하는 경우 관련 요청 및 응답은 공용 네트워크 또는 인터넷을 통해 전송되지 않습니다.

{{% alert %}}
보안 사설 연결은 곧 전용 클라우드의 고급 보안 옵션으로 제공될 예정입니다.
{{% /alert %}}

보안 사설 연결은 AWS, GCP 및 Azure의 전용 클라우드 인스턴스에서 사용할 수 있습니다.

* AWS의 [AWS Privatelink](https://aws.amazon.com/privatelink/) 사용
* GCP의 [GCP Private Service Connect](https://cloud.google.com/vpc/docs/private-service-connect) 사용
* Azure의 [Azure Private Link](https://learn.microsoft.com/en-us/azure/private-link/private-link-overview) 사용

활성화되면 W&B는 인스턴스에 대한 사설 엔드포인트 서비스를 생성하고 연결할 관련 DNS URI를 제공합니다. 이를 통해 클라우드 계정에서 사설 엔드포인트를 생성하여 관련 트래픽을 사설 엔드포인트 서비스로 라우팅할 수 있습니다. 사설 엔드포인트는 클라우드 VPC 또는 VNet 내에서 실행되는 AI 트레이닝 워크로드에 대해 더 쉽게 설정할 수 있습니다. 사용자 브라우저에서 W&B 앱 UI로의 트래픽에 대해 동일한 메커니즘을 사용하려면 회사 네트워크에서 클라우드 계정의 사설 엔드포인트로 적절한 DNS 기반 라우팅을 구성해야 합니다.

{{% alert %}}
이 기능을 사용하려면 W&B 팀에 문의하십시오.
{{% /alert %}}

[IP 허용 목록]({{< relref path="./ip-allowlisting.md" lang="ko" >}})과 함께 보안 사설 연결을 사용할 수 있습니다. IP 허용 목록에 보안 사설 연결을 사용하는 경우 W&B는 AI 워크로드의 모든 트래픽과 가능한 경우 사용자 브라우저의 대부분 트래픽에 대해 보안 사설 연결을 사용하고 권한 있는 위치에서 인스턴스 관리에 IP 허용 목록을 사용하는 것이 좋습니다.
