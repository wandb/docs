---
title: Prodigy
description: W&B와 Prodigy를 통합하는 방법
menu:
  default:
    identifier: ko-guides-integrations-prodigy
    parent: integrations
weight: 290
---

[Prodigy](https://prodi.gy/)는 머신러닝 모델 트레이닝과 평가 데이터, 오류 분석, 데이터 검사 및 정제를 위한 주석 툴입니다. [W&B Tables]({{< relref path="/guides/models/tables/tables-walkthrough.md" lang="ko" >}})을 사용하면 W&B 내에서 데이터셋을 손쉽게 로그하고, 시각화하며, 분석하고, 공유할 수 있습니다(이 외에도 다양한 활용이 가능합니다!).

[W&B와 Prodigy의 인테그레이션](https://github.com/wandb/wandb/blob/master/wandb/integration/prodigy/prodigy.py)은 Prodigy로 주석된 데이터셋을 W&B에 바로 업로드하여 Tables에서 활용할 수 있게 해주는 심플하고 사용이 쉬운 기능을 제공합니다.

아래 몇 줄의 코드만으로 바로 사용하실 수 있습니다:

```python
import wandb
from wandb.integration.prodigy import upload_dataset

with wandb.init(project="prodigy"):
    upload_dataset("news_headlines_ner")
```

이렇게 하면 아래와 같이 시각적이고, 상호작용 가능하며, 공유가 쉬운 테이블을 바로 확인할 수 있습니다:

{{< img src="/images/integrations/prodigy_interactive_visual.png" alt="Prodigy annotation table" >}}

## 퀵스타트

`wandb.integration.prodigy.upload_dataset`를 사용하면, 주석이 달린 Prodigy 데이터셋을 로컬 Prodigy 데이터베이스로부터 W&B에 바로 업로드하고 [Table]({{< relref path="/ref/python/sdk/data-types/table.md" lang="ko" >}}) 포맷으로 변환할 수 있습니다. Prodigy의 설치 및 설정 등 자세한 내용은 [Prodigy 문서](https://prodi.gy/docs/)를 참고하세요.

W&B는 이미지와 명명된 엔티티 필드를 각각 [`wandb.Image`]({{< relref path="/ref/python/sdk/data-types/image.md" lang="ko" >}}), [`wandb.Html`]({{< relref path="/ref/python/sdk/data-types/html.md" lang="ko" >}})로 자동 변환하려 시도합니다. 이처럼 시각화된 결과를 포함하기 위해 결과 테이블에 추가 컬럼이 생성될 수 있습니다.

## 자세한 예제 살펴보기

W&B Prodigy 인테그레이션으로 생성된 예시 시각화를 보고 싶다면 [Visualizing Prodigy Datasets Using W&B Tables](https://wandb.ai/kshen/prodigy/reports/Visualizing-Prodigy-Datasets-Using-W-B-Tables--Vmlldzo5NDE2MTc)를 참고해보세요.  

## spaCy도 함께 사용 중이신가요?

W&B는 spaCy와의 인테그레이션도 제공합니다. [여기에서 docs를 확인하세요]({{< relref path="/guides/integrations/spacy" lang="ko" >}}).