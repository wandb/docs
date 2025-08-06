---
title: 사전 준비 사항
description: W&B Inference를 사용하기 위한 환경 설정
linkTitle: Prerequisites
menu:
  default:
    identifier: ko-guides-inference-prerequisites
weight: 1
---

W&B Inference 서비스를 API 또는 UI를 통해 사용하기 전에 아래 단계를 완료하세요.

{{< alert title="Tip" >}}
시작하기 전에 [이용 안내 및 제한 사항]({{< relref path="usage-limits" lang="ko" >}})을 검토하여 비용과 제약 조건을 확인하세요.
{{< /alert >}}

## W&B 계정 및 프로젝트 설정하기

W&B Inference를 사용하려면 다음 항목이 필요합니다.

1. **W&B 계정**  
   [W&B](https://app.wandb.ai/login?signup=true)에서 가입하세요.

2. **W&B API 키**  
   [https://wandb.ai/authorize](https://wandb.ai/authorize)에서 API 키를 발급받으세요.

3. **W&B 프로젝트**  
   W&B 계정 내에서 프로젝트를 생성하여 사용 현황을 추적하세요.

## 환경 설정하기 (Python)

Python에서 Inference API를 사용하려면 다음도 필요합니다.

1. 위의 일반 요구 사항을 완료하세요.

2. 필요한 라이브러리를 설치하세요:

   ```bash
   pip install openai weave
   ```

{{< alert title="Note" >}}
`weave` 라이브러리는 선택 사항이지만 추천합니다. LLM 애플리케이션의 추적이 가능합니다. 자세한 내용은 [Weave 퀵스타트]({{< relref path="../quickstart" lang="ko" >}})를 참고하세요.

Weave와 함께 W&B Inference를 사용하는 코드 샘플은 [사용 예시]({{< relref path="examples" lang="ko" >}})에서 확인할 수 있습니다.
{{< /alert >}}

## 다음 단계

사전 준비가 완료되면:

- 이용 가능한 엔드포인트는 [API reference]({{< relref path="api-reference" lang="ko" >}})에서 확인하세요.
- 서비스 작동 방식을 확인하려면 [사용 예시]({{< relref path="examples" lang="ko" >}})를 살펴보세요.
- 웹 인터페이스에서 모델에 엑세스하려면 [UI 가이드]({{< relref path="ui-guide" lang="ko" >}})를 참고하세요.