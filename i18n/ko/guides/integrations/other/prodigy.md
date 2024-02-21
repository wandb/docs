---
description: How to integrate W&B with Prodigy.
slug: /guides/integrations/prodigy
displayed_sidebar: default
---

# Prodigy

[Prodigy](https://prodi.gy/)는 머신 러닝 모델, 오류 분석, 데이터 검사 및 정리를 위한 학습 및 평가 데이터를 생성하는 주석 도구입니다. [W&B 테이블](../../tables/tables-walkthrough.md)은 W&B 내에서 데이터세트(그리고 그 이상의 것들!)를 로그, 시각화, 분석 및 공유할 수 있게 합니다.

[Prodigy와의 W&B 통합](https://github.com/wandb/wandb/blob/master/wandb/integration/prodigy/prodigy.py)은 테이블과 함께 사용하기 위해 Prodigy로 주석이 달린 데이터세트를 직접 W&B에 업로드하는 간단하고 사용하기 쉬운 기능을 추가합니다.

이러한 몇 줄의 코드를 실행하세요:

```python
import wandb
from wandb.integration.prodigy import upload_dataset

with wandb.init(project="prodigy"):
    upload_dataset("news_headlines_ner")
```

그러면 이와 같은 시각적이고 상호작용적이며 공유 가능한 테이블을 얻을 수 있습니다:

![](/images/integrations/prodigy_interactive_visual.png)

## 퀵스타트

`wandb.integration.prodigy.upload_dataset`을 사용하여 주석이 달린 Prodigy 데이터세트를 로컬 Prodigy 데이터베이스에서 직접 W&B의 [테이블](https://docs.wandb.ai/ref/python/data-types/table) 형식으로 업로드하세요. Prodigy에 대한 자세한 정보, 설치 및 설정을 포함하여 [Prodigy 문서](https://prodi.gy/docs/)를 참조하세요.

W&B는 자동으로 이미지와 명명된 엔티티 필드를 각각 [`wandb.Image`](https://docs.wandb.ai/ref/python/data-types/image)와 [`wandb.Html`](https://docs.wandb.ai/ref/python/data-types/html)로 변환하려고 시도합니다. 이러한 시각화를 포함하기 위해 결과 테이블에 추가 열이 추가될 수 있습니다.

## 자세한 예제 읽어보기

W&B Prodigy 통합으로 생성된 예제 시각화를 탐색해보려면 [W&B 테이블을 사용하여 Prodigy 데이터세트 시각화하기](https://wandb.ai/kshen/prodigy/reports/Visualizing-Prodigy-Datasets-Using-W-B-Tables--Vmlldzo5NDE2MTc)를 확인하세요.

## spaCy도 사용하고 있나요?

W&B는 spaCy와도 통합되어 있으며, [여기서 문서](https://docs.wandb.ai/guides/integrations/spacy)를 확인할 수 있습니다.