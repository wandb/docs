---
title: 사용 정보 및 한도
description: W&B Inference의 요금제, 사용 한도, 계정 제한 사항 알아보기
linkTitle: Usage & Limits
menu:
  default:
    identifier: ko-guides-inference-usage-limits
weight: 20
---

W&B Inference를 사용하기 전에 가격, 제한 사항, 그리고 기타 중요한 사용 안내를 확인하세요.

## 가격

자세한 모델 가격 정보는 [W&B Inference 가격 안내](https://wandb.ai/site/pricing/inference)를 참고하세요.

## 크레딧 추가 구매

W&B Inference 크레딧은 Free, Pro, Academic 플랜에 한정 기간 제공됩니다. Enterprise의 크레딧 제공 여부는 다를 수 있습니다. 크레딧이 모두 소진되면:

- **Free 계정**은 계속 사용하려면 유료 플랜으로 업그레이드해야 합니다. [Pro 또는 Enterprise로 업그레이드](https://wandb.ai/subscriptions)
- **Pro 플랜 사용자**는 [모델별 가격 정책](https://wandb.ai/site/pricing/inference)에 따라 월별로 추가 사용량이 청구됩니다.
- **Enterprise 계정**은 담당 영업대표에게 문의하세요.

## 계정 등급별 기본 사용 한도

각 계정 등급에는 비용 관리 및 예상치 못한 청구를 방지하기 위한 기본 한도가 설정되어 있습니다. W&B는 유료 Inference 엑세스에 대해 선불 결제를 요구합니다.

사용자에 따라 한도를 변경해야 할 수 있습니다. 한도 조정이 필요하면 담당 영업대표 또는 지원팀에 문의하세요.

| 계정 등급      | 기본 한도        | 한도 변경 방법                         |
|----------------|------------------|----------------------------------------|
| Pro            | $6,000/월        | 담당 영업대표 또는 지원팀에 문의하여 수동 심사 요청 |
| Enterprise     | $700,000/년      | 담당 영업대표 또는 지원팀에 문의하여 수동 심사 요청 |

## 동시 요청 제한

요청 속도를 초과하면, API는 `429 Concurrency limit reached for requests` 응답을 반환합니다. 이 오류를 해결하려면 동시에 보내는 요청 수를 줄이세요. 자세한 문제 해결 방법은 [W&B Inference 지원 문서](/support/inference/)를 참고하세요.

W&B는 W&B Project별로 속도 제한을 적용합니다. 예를 들어, 한 팀에 3개의 Project가 있다면 각 Project마다 독립적인 제한 쿼터가 있습니다.

## Personal entity 미지원

{{< alert title="안내" >}}
W&B는 2024년 5월부로 personal entity를 더 이상 지원하지 않습니다. 해당 내용은 기존(legacy) 계정에만 적용됩니다.
{{< /alert >}}

Personal 계정(personal entities)에서는 W&B Inference를 사용할 수 없습니다. W&B Inference를 사용하려면 Team을 생성하여 personal 계정이 아닌 유형으로 전환하세요.

## 지리적 제한

Inference 서비스는 지원되는 특정 지역에서만 엑세스할 수 있습니다. 자세한 내용은 [서비스 약관](https://docs.coreweave.com/docs/policies/terms-of-service/terms-of-use#geographic-restrictions)을 참고하세요.

## 다음 단계

- 시작 전에 [사전 요구사항]({{< relref path="prerequisites" lang="ko" >}})을 확인하세요.
- [사용 가능한 models]({{< relref path="models" lang="ko" >}}) 및 각 모델의 비용을 확인하세요.