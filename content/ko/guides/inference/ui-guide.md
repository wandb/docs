---
title: UI 가이드
description: 웹 인터페이스를 통해 W&B Inference 모델에 엑세스하세요.
menu:
  default:
    identifier: ko-guides-inference-ui-guide
weight: 60
---

W&B Inference 서비스를 웹 UI를 통해 사용하는 방법을 알아보세요. UI 사용 전 [사전 준비 사항]({{< relref path="prerequisites" lang="ko" >}})을 완료해 주세요.

## Inference 서비스 엑세스하기

Inference 서비스는 세 가지 경로를 통해 엑세스할 수 있습니다.

### 직접 링크

[https://wandb.ai/inference](https://wandb.ai/inference)로 이동하세요.

### Inference 탭에서

1. [https://wandb.ai/](https://wandb.ai/)에서 본인의 W&B 계정으로 이동합니다.
2. 왼쪽 사이드바에서 **Inference**를 선택합니다.
3. 사용 가능한 모델과 모델 정보가 포함된 페이지가 나타납니다.

{{< img src="/images/inference/inference-playground-single.png" alt="Playground에서 Inference 모델 사용 예시" >}}

### Playground 탭에서

1. 왼쪽 사이드바에서 **Playground**를 선택하세요. Playground 채팅 UI가 나타납니다.
2. LLM 드롭다운 리스트에서 **W&B Inference** 위에 마우스를 올려놓으세요. 오른쪽에 사용 가능한 모델 목록이 포함된 드롭다운이 나타납니다.
3. 모델 드롭다운에서 다음과 같은 작업이 가능합니다:
   - 원하는 모델 이름을 클릭하여 [Playground에서 바로 사용](#try-a-model-in-the-playground)
   - [여러 모델을 비교](#compare-multiple-models)

{{< img src="/images/inference/inference-playground.png" alt="Playground의 Inference 모델 드롭다운" >}}

## Playground에서 모델 사용해 보기

[모델 선택](#access-the-inference-service) 후, Playground에서 테스트할 수 있습니다. 다음과 같은 작업을 할 수 있습니다:

- [모델 설정 및 파라미터 커스터마이즈](https://weave-docs.wandb.ai/guides/tools/playground#customize-settings)
- [메시지 추가, 재시도, 편집 및 삭제](https://weave-docs.wandb.ai/guides/tools/playground#message-controls)
- [커스텀 설정을 가진 모델 저장 및 재사용](https://weave-docs.wandb.ai/guides/tools/playground#saved-models)
- [여러 모델 비교](#compare-multiple-models)

## 여러 모델 비교하기

Playground에서 여러 Inference 모델을 나란히 비교할 수 있습니다. Compare 뷰는 다음 두 경로로 들어갈 수 있습니다.

### Inference 탭에서

1. 왼쪽 사이드바에서 **Inference**를 선택합니다. 사용 가능한 모델 페이지가 나타납니다.
2. 비교하고 싶은 모델 카드(모델 이름 제외)를 아무 곳이나 클릭하여 선택하세요. 카드 테두리가 파란색으로 바뀝니다.
3. 비교를 원하는 각 모델마다 반복합니다.
4. 선택한 카드 중 아무 카드에서나 **Compare N models in the Playground**를 클릭하세요. `N`은 선택된 모델 개수로 표시됩니다.
5. 비교 화면이 열립니다.

이제 [Playground에서 모델 사용해 보기](#try-a-model-in-the-playground)의 모든 기능을 활용하며 모델을 비교할 수 있습니다.

{{< img src="/images/inference/inference-playground-compare.png" alt="Playground에서 여러 모델을 비교 선택" >}}

### Playground 탭에서

1. 왼쪽 사이드바에서 **Playground**를 선택하세요. Playground 채팅 UI가 나타납니다.
2. LLM 드롭다운 리스트에서 **W&B Inference** 위에 마우스를 올려놓으면 모델 드롭다운이 오른쪽에 나타납니다.
3. 드롭다운에서 **Compare**를 선택하세요. **Inference** 탭이 표시됩니다.
4. 비교하고 싶은 모델 카드(모델 이름 제외)를 아무 곳이나 클릭하면 카드 테두리가 파란색으로 변경됩니다.
5. 비교할 각 모델마다 위 동작을 반복합니다.
6. 선택한 카드 중 아무 카드에서나 **Compare N models in the Playground**를 클릭하면 비교 화면이 열립니다.

이제 [Playground에서 모델 사용해 보기](#try-a-model-in-the-playground)의 모든 기능을 활용하여 모델을 비교할 수 있습니다.

## 결제 및 사용 정보 확인

조직 관리자라면 W&B UI에서 남은 크레딧, 사용 내역, 다가오는 결제 금액을 확인할 수 있습니다.

1. UI 내 W&B **Billing** 페이지로 이동합니다.
2. 오른쪽 하단에서 Inference 결제 정보 카드를 찾으세요.
3. 여기서 다음이 가능합니다:
   - **View usage**를 클릭하여 전체 사용 내역을 확인
   - 곧 청구될 Inference 이용 요금 확인 (유료 플랜의 경우)

{{< alert title="Tip" >}}
모델별 요금 세부 정보는 [Inference 요금 페이지](https://wandb.ai/site/pricing/inference)에서 확인하세요.
{{< /alert >}}

## 다음 단계

- [사용 가능한 모델]({{< relref path="models" lang="ko" >}})을 검토하여 필요에 맞는 모델을 찾아보세요.
- 프로그램을 통해 엑세스하려면 [API]({{< relref path="api-reference" lang="ko" >}})를 시도해보세요.
- [사용 예시]({{< relref path="examples" lang="ko" >}})에서 코드 샘플을 확인하세요.