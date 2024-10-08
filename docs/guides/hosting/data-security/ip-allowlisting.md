---
title: Configure IP allowlisting for Dedicated Cloud
displayed_sidebar: default
---

당신의 [전용 클라우드](../hosting-options/dedicated_cloud.md) 인스턴스에 엑세스할 수 있는 IP 어드레스 목록을 허가된 목록으로만 제한할 수 있습니다. 이것은 W&B API에 대한 AI 작업 부하의 엑세스와 W&B 앱 UI에 대한 사용자 브라우저의 엑세스 모두에 적용됩니다. 전용 클라우드 인스턴스에 대해 IP 허용 목록이 설정되면, W&B는 다른 허가되지 않은 위치에서의 모든 요청을 거부합니다. 전용 클라우드 인스턴스에 대한 IP 허용 목록 구성을 위해 W&B 팀에 문의하십시오.

IP 허용 목록은 AWS, GCP, Azure의 전용 클라우드 인스턴스에서 사용할 수 있습니다.

[보안 사설 연결](./private-connectivity.md)과 함께 IP 허용 목록을 사용할 수 있습니다. IP 허용 목록을 보안 사설 연결과 함께 사용하는 경우, W&B는 가능하다면 모든 AI 작업 부하의 트래픽과 사용자 브라우저의 대부분의 트래픽에 대해 보안 사설 연결을 사용하고, 특정 위치에서의 인스턴스 관리에 대해 IP 허용 목록을 사용할 것을 권장합니다.

:::important
W&B는 개별 `/32` IP 어드레스보다는 회사 또는 비즈니스 이그레스 게이트웨이에 할당된 [CIDR블록](https://en.wikipedia.org/wiki/Classless_Inter-Domain_Routing)을 사용할 것을 강력하게 권장합니다. 개별 IP 어드레스를 사용하는 것은 확장성이 없으며 클라우드 당 엄격한 제한이 있습니다.
:::