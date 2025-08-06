---
title: 전용 클라우드에 대한 IP 허용 목록 구성
menu:
  default:
    identifier: ko-guides-hosting-data-security-ip-allowlisting
    parent: data-security
weight: 3
---

[Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}}) 인스턴스에 대해, 허가된 IP 어드레스 리스트에서만 엑세스할 수 있도록 제한할 수 있습니다. 이 제한은 AI 워크로드에서 W&B API로의 엑세스와, 사용자 브라우저에서 W&B 앱 UI로의 엑세스 모두에 적용됩니다. Dedicated Cloud 인스턴스에 IP 허용리스트가 설정되면, W&B는 허가되지 않은 위치에서의 모든 요청을 거부합니다. Dedicated Cloud 인스턴스에 대한 IP 허용리스트 설정을 원하시면 W&B 팀에 문의해 주세요.

IP 허용리스트 기능은 AWS, GCP, Azure 상의 Dedicated Cloud 인스턴스에서 지원됩니다.

[보안 프라이빗 연결]({{< relref path="./private-connectivity.md" lang="ko" >}})과 함께 IP 허용리스트를 사용할 수 있습니다. 보안 프라이빗 연결과 IP 허용리스트를 같이 사용하는 경우, W&B에서는 AI 워크로드에서 발생하는 모든 트래픽과 가능하다면 사용자 브라우저의 대부분 트래픽에 보안 프라이빗 연결을 이용하고, 인스턴스 관리 목적으로는 IP 허용리스트를 권한 있는 위치에서만 사용하실 것을 권장합니다.

{{% alert color="secondary" %}}
W&B에서는 개별 `/32` IP 어드레스보다는, 기업 또는 비즈니스 egress 게이트웨이에 할당된 [CIDR 블록](https://en.wikipedia.org/wiki/Classless_Inter-Domain_Routing) 사용을 강력히 권장합니다. 개별 IP 어드레스를 사용하는 것은 확장성이 부족하고 각 클라우드별로 엄격한 제한이 있습니다.
{{% /alert %}}