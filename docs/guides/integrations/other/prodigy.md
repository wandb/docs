---
title: Prodigy
description: W&B를 Prodigy와 통합하는 방법.
slug: /guides/integrations/prodigy
displayed_sidebar: default
---

[Prodigy](https://prodi.gy/)는 기계학습 모델, 오류 분석, 데이터 검사 및 정리를 위한 트레이닝 및 평가 데이터를 생성하는 어노테이션 툴입니다. [W&B Tables](../../tables/tables-walkthrough.md)를 사용하면 W&B 내에서 데이터셋 (및 그 이상)을 로그, 시각화, 분석, 공유할 수 있습니다.

[Prodigy와의 W&B 인테그레이션](https://github.com/wandb/wandb/blob/master/wandb/integration/prodigy/prodigy.py)을 통해 Prodigy로 어노테이션된 데이터셋을 직접 W&B로 업로드하여 Tables에서 사용할 수 있는 간단하고 사용하기 쉬운 기능을 추가합니다.

다음과 같은 몇 줄의 코드를 실행하세요:

```python
import wandb
from wandb.integration.prodigy import upload_dataset

with wandb.init(project="prodigy"):
    upload_dataset("news_headlines_ner")
```

그리고 다음과 같은 시각적, 상호작용 가능한, 공유 가능한 테이블을 얻으세요:

![](/images/integrations/prodigy_interactive_visual.png)

## 퀵스타트

`wandb.integration.prodigy.upload_dataset`를 사용하여 로컬 Prodigy 데이터베이스에서 직접 W&B로 어노테이션된 Prodigy 데이터셋을 [Table](/ref/python/data-types/table) 형식으로 업로드하세요. Prodigy에 대한 설치 및 설정을 포함한 더 많은 정보는 [Prodigy documentation](https://prodi.gy/docs/)을 참조하세요.

W&B는 자동으로 이미지를 [`wandb.Image`](/ref/python/data-types/image) 및 명명된 엔티티 필드를 [`wandb.Html`](/ref/python/data-types/html)로 변환하려고 시도합니다. 추가 컬럼이 시각화를 포함하도록 결과 테이블에 추가될 수도 있습니다.

## 상세 예제 읽기

W&B Prodigy 인테그레이션으로 생성된 시각화를 예제로 탐색하려면 [Visualizing Prodigy Datasets Using W&B Tables](https://wandb.ai/kshen/prodigy/reports/Visualizing-Prodigy-Datasets-Using-W-B-Tables--Vmlldzo5NDE2MTc)를 참고하세요.  

## spaCy도 사용 중이신가요?

W&B는 spaCy와의 인테그레이션도 제공하니, [docs here](/guides/integrations/spacy)를 참조하세요.