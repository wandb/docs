---
title: Example tables
description: W&B 테이블의 예시
displayed_sidebar: default
---

다음 섹션에서는 테이블을 사용하는 몇 가지 방법을 강조합니다:

### 데이터 보기

모델 트레이닝 또는 평가 중 메트릭 및 다양한 미디어를 로그한 후, 지속적으로 클라우드 또는 [호스팅 인스턴스](/guides/hosting)에 동기화된 데이터베이스에서 결과를 시각화합니다.

![데이터의 예제를 보고 데이터의 수와 분포를 검증하세요](/images/data_vis/tables_see_data.png)

예를 들어, 이 테이블은 [사진 데이터셋의 균형 잡힌 분할](https://wandb.ai/stacey/mendeleev/artifacts/balanced_data/inat_80-10-10_5K/ab79f01e007113280018/files/data_split.table.json)을 보여줍니다.

### 데이터를 대화형으로 탐색하기

정적 파일을 검색하거나 분석 스크립트를 다시 실행하지 않고도 테이블을 보고, 정렬하고, 필터링하고, 그룹화하고, 연결하고 질의하여 데이터 및 모델 성능을 이해하세요.

![원곡과 그 합성 버전(타임브르 전이 포함)을 들으세요](/images/data_vis/explore_data.png)

예를 들어, [스타일 변환된 오디오](https://wandb.ai/stacey/cshanty/reports/Whale2Song-W-B-Tables-for-Audio--Vmlldzo4NDI3NzM)에 대한 이 리포트를 참조하세요.

### 모델 버전 비교하기

다양한 트레이닝 에포크, 데이터셋, 하이퍼파라미터 선택, 모델 아키텍처 등에 걸친 결과를 빠르게 비교하세요.

![세부적인 차이를 보세요: 왼쪽 모델은 일부 빨간색 인도를 감지하고, 오른쪽 모델은 감지하지 않습니다.](/images/data_vis/compare_model_versions.png)

예를 들어, [동일한 테스트 이미지에서 두 모델 비교](https://wandb.ai/stacey/evalserver_answers_2/artifacts/results/eval_Daenerys/c2290abd3d7274f00ad8/files/eval_results.table.json#b6dae62d4f00d31eeebf$eval_Bob)를 보여주는 이 테이블을 참조하세요.

### 모든 세부 사항을 추적하고 더 큰 그림을 이해하기

특정 단계에서의 특정 예측을 시각화하기 위해 확대하고, 집계된 통계를 보기 위해 축소하세요. 오류 패턴을 식별하고 개선 기회를 이해합니다. 이 툴은 단일 모델 트레이닝의 특정 단계 또는 다양한 모델 버전에 걸친 결과를 비교하는 데 유용합니다.

![](/images/data_vis/track_details.png)

예를 들어, [MNIST 데이터셋에서 한 에포크 후 그리고 다섯 에포크 후 결과 분석](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json#7dd0cd845c0edb469dec)을 보여주는 이 예제 테이블을 참조하세요.

## W&B 테이블을 사용한 예제 프로젝트

다음은 W&B 테이블을 사용하는 실제 W&B 프로젝트를 강조합니다.

### 이미지 분류

[CNN이 iNaturalist 사진에서 10종의 생물(식물, 새, 곤충 등)을 어떻게 식별하는지 보려면 이 리포트를 읽어보세요](https://wandb.ai/stacey/mendeleev/reports/Visualize-Data-for-Image-Classification--VmlldzozNjE3NjA) 또는 [이 Colab](https://wandb.me/dsviz-nature-colab)을 따르세요, 또는 이 [Artifacts의 문맥](https://wandb.ai/stacey/mendeleev/artifacts/val_epoch_preds/val_pred_gawf9z8j/2dcee8fa22863317472b/files/val_epoch_res.table.json)을 탐색하세요.

![두 모델의 예측에 따른 실제 레이블의 분포 비교](/images/data_vis/image_classification.png)

### 오디오

타임브르 전이에 관한 [이 리포트에서](https://wandb.ai/stacey/cshanty/reports/Whale2Song-W-B-Tables-for-Audio--Vmlldzo4NDI3NzM) 오디오 테이블을 상호작용하세요. 녹음된 고래 소리와 악기(바이올린, 트럼펫 등)로 같은 멜로디를 합성한 버전을 비교할 수 있습니다. [이 Colab](http://wandb.me/audio-transfer)으로 자신의 노래를 녹음하고 합성 버전을 W&B에서 탐색할 수도 있습니다.

![](/images/data_vis/audio.png)

### 텍스트

트레이닝 데이터나 생성된 출력물의 텍스트 샘플을 탐색하고, 관련 필드로 동적으로 그룹화하며, 모델 변형 또는 실험 설정에 따라 평가를 정렬합니다. 텍스트를 마크다운으로 렌더링하거나 시각적 차이 모드를 사용하여 텍스트를 비교하세요. [이 리포트에서](https://wandb.ai/stacey/nlg/reports/Visualize-Text-Data-Predictions--Vmlldzo1NzcwNzY) 셰익스피어를 생성하기 위한 간단한 문자 기반 RNN을 탐색합니다.

![숨겨진 레이어의 크기를 두 배로 늘리면 더 창의적인 프롬프트 완성이 가능합니다.](/images/data_vis/shakesamples.png)

### 비디오

트레이닝 중에 기록된 비디오를 탐색하고 집계하여 모델을 이해합니다. 여기 [SafeLife 벤치마크](https://wandb.ai/safelife/v1dot2/benchmark)를 사용하여 [부수 효과 최소화를 목표로 하는 RL 에이전트](https://wandb.ai/stacey/saferlife/artifacts/video/videos_append-spawn/c1f92c6e27fa0725c154/files/video_examples.table.json)를 위한 초기 예제가 있습니다.

![성공적인 에이전트 몇 개를 쉽게 탐색하세요](/images/data_vis/video.png)

### 표 형식 데이터

버전 제어와 중복 제거를 통해 [표 형식 데이터를 분할하고 전처리하는 방법](https://wandb.ai/dpaiton/splitting-tabular-data/reports/Tabular-Data-Versioning-and-Deduplication-with-Weights-Biases--VmlldzoxNDIzOTA1)에 대한 보고서를 확인하세요.

![테이블과 Artifacts가 함께 버전 제어, 라벨링 및 데이터셋 반복의 중복 제거를 수행합니다](/images/data_vis/tabs.png)

### 모델 변형 비교(시멘틱 세그멘테이션)

시멘틱 세그멘테이션에 테이블을 로그하고 다양한 모델을 비교하기 위한 [인터랙티브 노트북](https://wandb.me/dsviz-cars-demo) 및 [실시간 예제](https://wandb.ai/stacey/evalserver_answers_2/artifacts/results/eval_Daenerys/c2290abd3d7274f00ad8/files/eval_results.table.json#a57f8e412329727038c2$eval_Ada)를 제공. [이 테이블에서](https://wandb.ai/stacey/evalserver_answers_2/artifacts/results/eval_Daenerys/c2290abd3d7274f00ad8/files/eval_results.table.json) 자신의 쿼리를 시도해 보세요.

![동일한 테스트 세트에서 두 모델의 예측 중 최고의 결과 찾기](/images/data_vis/comparing_model_variants.png)

### 트레이닝 시간 동안의 개선 분석

[시간에 따른 예측 시각화](https://wandb.ai/stacey/mnist-viz/reports/Visualize-Predictions-over-Time--Vmlldzo1OTQxMTk)에 대한 상세 보고서와 동반되는 [인터랙티브 노트북](https://wandb.me/dsviz-mnist-colab).