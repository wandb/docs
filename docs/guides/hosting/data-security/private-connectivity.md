---
title: Configure private connectivity to Dedicated Cloud
displayed_sidebar: default
---

[Dedicated Cloud](../hosting-options/dedicated_cloud.md) 인스턴스에 클라우드 제공자의 안전한 사설 네트워크를 통해 연결할 수 있습니다. 이는 AI 워크로드에서 W&B API로의 엑세스와 선택적으로 사용자 브라우저에서 W&B 앱 UI로의 엑세스에 적용됩니다. 사설 연결을 사용할 때 관련 요청과 응답은 공용 네트워크나 인터넷을 통과하지 않습니다.

:::info
안전한 사설 연결은 전용 클라우드에서 고급 보안 옵션으로 미리보기로 제공됩니다.
:::

안전한 사설 연결은 AWS, GCP 및 Azure의 전용 클라우드 인스턴스에서 사용할 수 있습니다:

* AWS에서 [AWS Privatelink](https://aws.amazon.com/privatelink/) 사용
* GCP에서 [GCP Private Service Connect](https://cloud.google.com/vpc/docs/private-service-connect) 사용
* Azure에서 [Azure Private Link](https://azure.microsoft.com/en-us/products/private-link) 사용

활성화되면, W&B는 인스턴스를 위한 사설 엔드포인트 서비스를 생성하고, 연결에 필요한 관련 DNS URI를 제공합니다. 이를 통해 클라우드 계정 내에 사설 엔드포인트를 생성하여 관련 트래픽을 사설 엔드포인트 서비스로 라우팅할 수 있습니다. 사설 엔드포인트는 클라우드 VPC 또는 VNet 내에서 실행되는 AI 트레이닝 워크로드에 대해 설정하기가 더 쉽습니다. 사용자 브라우저의 트래픽을 W&B 앱 UI로 라우팅하기 위해 동일한 메커니즘을 사용하려면, 회사 네트워크에서 클라우드 계정의 사설 엔드포인트로의 적절한 DNS 기반 라우팅을 구성해야 합니다.

:::info
이 기능을 사용하려면, W&B 팀에 문의하십시오.
:::

[IP 허용 목록](./ip-allowlisting.md)과 함께 안전한 사설 연결을 사용할 수 있습니다. IP 허용 목록을 위한 사설 연결을 사용하는 경우, W&B는 AI 워크로드의 모든 트래픽과 사용자 브라우저의 대부분의 트래픽에 대해 사설 연결을 최대한 확보하면서, 특권이 있는 위치에서 인스턴스 관리를 위한 IP 허용 목록을 사용하는 것을 권장합니다.