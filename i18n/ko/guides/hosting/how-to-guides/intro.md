---
description: W&B Server Install
slug: /guides/hosting/how-to-guides
displayed_sidebar: default
---

# 설치

## W&B 프로덕션 및 개발

이 페이지에서는 W&B 서버를 설치할 수 있는 모든 방법을 찾을 수 있습니다.

사용 가능한 프로덕션 설치 유형은 다음과 같습니다:

- [AWS](./aws-tf.md)
- [Azure](./azure-tf.md)
- [GCP](./gcp-tf.md)
- [베어 메탈](./bare-metal.md)

클라우드 배포의 경우, W&B 서버를 안정적으로 실행하기 위해 필요한 모든 인프라 구성 요소를 프로비저닝하기 위해 [Terraform](https://developer.hashicorp.com/terraform/intro) 툴에 의존합니다.

:::info
Terraform에서 사용 가능한 [원격 백엔드](https://developer.hashicorp.com/terraform/language/settings/backends/configuration) 중 하나를 선택하여 [상태 파일](https://developer.hashicorp.com/terraform/language/state)을 저장하는 것이 좋습니다.

상태 파일은 모든 구성 요소를 다시 생성하지 않고도 업그레이드를 진행하거나 배포에서 변경을 수행하는 데 필요한 자원입니다.
:::

사용자가 전체 인프라를 프로비저닝하지 않고도 Weights and Biases 서버를 시도할 수 있도록 W&B 서버를 로컬에서 실행하는 것이 가능합니다.

- [개발 설정](./basic-setup.md)

이 모드는 **프로덕션에는 권장되지 않습니다**