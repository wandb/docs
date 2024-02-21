---
description: W&B Server Install
slug: /guides/hosting/how-to-guides
displayed_sidebar: default
---

# 설치

## W&B 프로덕션 및 개발

이 페이지에서는 W&B 서버를 설치하는 모든 가능한 방법을 찾을 수 있습니다.

사용 가능한 프로덕션 설치 유형은 다음과 같습니다:

- [AWS](./aws-tf.md)
- [Azure](./azure-tf.md)
- [GCP](./gcp-tf.md)
- [베어 메탈](./bare-metal.md)

클라우드 배포의 경우, W&B 서버를 안정적으로 실행하는 데 필요한 모든 인프라 구성 요소를 프로비저닝하기 위해 [Terraform](https://developer.hashicorp.com/terraform/intro)이라는 도구에 의존합니다.

:::info
Terraform을 사용하여 [상태 파일](https://developer.hashicorp.com/terraform/language/state)을 저장하기 위해 사용 가능한 [리모트 백엔드](https://developer.hashicorp.com/terraform/language/settings/backends/configuration) 중 하나를 선택하는 것이 좋습니다.

상태 파일은 배포에서 모든 구성 요소를 다시 만들지 않고 업그레이드를 진행하거나 변경사항을 적용하는 데 필요한 자원입니다.
:::

전체 인프라를 프로비저닝할 필요 없이 사용자가 Weights and Biases 서버를 직접 시도할 수 있도록 W&B 서버를 로컬에서 실행하는 것이 가능합니다.

- [개발 설정](./basic-setup.md)

이 모드는 **프로덕션에는 권장되지 않습니다**