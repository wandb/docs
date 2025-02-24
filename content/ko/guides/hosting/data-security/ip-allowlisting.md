---
title: Configure IP allowlisting for Dedicated Cloud
menu:
  default:
    identifier: ko-guides-hosting-data-security-ip-allowlisting
    parent: data-security
weight: 3
---

IP 주소 허용 목록을 사용하여 허가된 IP 주소 목록에서만 [전용 클라우드]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}}) 인스턴스에 대한 엑세스를 제한할 수 있습니다. 이는 AI 워크로드에서 W&B API로의 엑세스와 사용자 브라우저에서 W&B 앱 UI로의 엑세스에도 적용됩니다. 전용 클라우드 인스턴스에 대해 IP 허용 목록이 설정되면 W&B는 다른 권한 없는 위치에서 오는 모든 요청을 거부합니다. 전용 클라우드 인스턴스에 대한 IP 허용 목록을 구성하려면 W&B 팀에 문의하세요.

IP 허용 목록은 AWS, GCP 및 Azure의 전용 클라우드 인스턴스에서 사용할 수 있습니다.

[보안 사설 연결]({{< relref path="./private-connectivity.md" lang="ko" >}})과 함께 IP 허용 목록을 사용할 수 있습니다. 보안 사설 연결과 함께 IP 허용 목록을 사용하는 경우, AI 워크로드의 모든 트래픽과 가능한 경우 사용자 브라우저의 대부분의 트래픽에 대해 보안 사설 연결을 사용하고 권한 있는 위치에서 인스턴스 관리에 IP 허용 목록을 사용하는 것이 좋습니다.

{{% alert color="secondary" %}}
W&B는 개별 `/32` IP 어드레스 대신 회사 또는 비즈니스 이그레스 게이트웨이에 할당된 [CIDR 블록](https://en.wikipedia.org/wiki/Classless_Inter-Domain_Routing)을 사용하는 것을 강력히 권장합니다. 개별 IP 어드레스를 사용하는 것은 확장성이 떨어지며 클라우드당 엄격한 제한이 있습니다.
{{% /alert %}}
