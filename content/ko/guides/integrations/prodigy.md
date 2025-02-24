---
title: Prodigy
description: W&B와 Prodigy를 통합하는 방법.
menu:
  default:
    identifier: ko-guides-integrations-prodigy
    parent: integrations
weight: 290
---

[Prodigy](https://prodi.gy/) 는 기계 학습 모델, 오류 분석, 데이터 검사 및 정리 를 위한 트레이닝 및 평가 데이터를 생성하기 위한 어노테이션 툴입니다. [W&B Tables]({{< relref path="/guides/core/tables/tables-walkthrough.md" lang="ko" >}}) 를 사용하면 W&B 내에서 데이터셋 (및 그 이상!) 을 기록, 시각화, 분석 및 공유할 수 있습니다.

[Prodigy 와의 W&B integration](https://github.com/wandb/wandb/blob/master/wandb/integration/prodigy/prodigy.py) 는 간단하고 사용하기 쉬운 기능을 추가하여 Prodigy 로 어노테이션된 데이터셋을 Tables 와 함께 사용하기 위해 W&B 에 직접 업로드할 수 있습니다.

다음과 같은 몇 줄의 코드를 실행합니다.

```python
import wandb
from wandb.integration.prodigy import upload_dataset

with wandb.init(project="prodigy"):
    upload_dataset("news_headlines_ner")
```

다음과 같은 시각적이고, 상호 작용이 가능하며, 공유 가능한 테이블을 얻으세요.

{{< img src="/images/integrations/prodigy_interactive_visual.png" alt="" >}}

## 퀵스타트

`wandb.integration.prodigy.upload_dataset` 을 사용하여 어노테이션된 Prodigy 데이터셋을 로컬 Prodigy 데이터베이스에서 [Table]({{< relref path="/ref/python/data-types/table" lang="ko" >}}) 형식으로 W&B 에 직접 업로드합니다. 설치 및 설정을 포함한 Prodigy 에 대한 자세한 내용은 [Prodigy documentation](https://prodi.gy/docs/) 을 참조하십시오.

W&B 는 이미지 및 명명된 엔터티 필드를 [`wandb.Image`]({{< relref path="/ref/python/data-types/image" lang="ko" >}}) 및 [`wandb.Html`]({{< relref path="/ref/python/data-types/html" lang="ko" >}}) 로 자동 변환하려고 시도합니다. 이러한 시각화를 포함하기 위해 결과 테이블에 추가 열이 추가될 수 있습니다.

## 자세한 예제 살펴보기

W&B Prodigy integration 으로 생성된 시각화 예제는 [Visualizing Prodigy Datasets Using W&B Tables](https://wandb.ai/kshen/prodigy/reports/Visualizing-Prodigy-Datasets-Using-W-B-Tables--Vmlldzo5NDE2MTc) 를 살펴보십시오.

## spaCy 도 사용하시나요?

W&B 에는 spaCy 와의 integration 도 있습니다. [여기에서 문서]({{< relref path="/guides/integrations/spacy" lang="ko" >}}) 를 참조하십시오.
