---
title: 예시 테이블
description: W&B Tables 예시
menu:
  default:
    identifier: ko-guides-models-tables-tables-gallery
    parent: tables
---

다음 섹션에서는 테이블을 다양하게 활용할 수 있는 방법을 소개합니다.

### 데이터 보기

모델 트레이닝이나 평가 중 메트릭과 다양한 미디어를 로그한 후, 클라우드에 동기화된 영구 데이터베이스나 [호스팅 인스턴스]({{< relref path="/guides/hosting" lang="ko" >}})에서 결과를 시각화할 수 있습니다.

{{< img src="/images/data_vis/tables_see_data.png" alt="데이터 탐색 테이블" max-width="90%" >}}

예를 들어, 이 테이블은 [사진 데이터셋의 균형 잡힌 분할 예시](https://wandb.ai/stacey/mendeleev/artifacts/balanced_data/inat_80-10-10_5K/ab79f01e007113280018/files/data_split.table.json)를 보여줍니다.

### 데이터 인터랙티브 탐색

테이블에서 직접 데이터를 조회, 정렬, 필터, 그룹화, 조인, 쿼리할 수 있어 데이터와 모델 성능을 쉽게 이해할 수 있습니다. 더 이상 정적인 파일을 일일이 찾거나 별도의 분석 스크립트를 매번 실행할 필요가 없습니다.

{{< img src="/images/data_vis/explore_data.png" alt="오디오 비교" max-width="90%">}}

예시로는 [스타일 변환이 적용된 오디오](https://wandb.ai/stacey/cshanty/reports/Whale2Song-W-B-Tables-for-Audio--Vmlldzo4NDI3NzM) 리포트를 참고하실 수 있습니다.

### 모델 버전 비교

서로 다른 트레이닝 에포크, Datasets, 하이퍼파라미터, 모델 아키텍처 등 다양한 조건에서 나온 결과를 빠르게 비교할 수 있습니다.

{{< img src="/images/data_vis/compare_model_versions.png" alt="모델 비교" max-width="90%">}}

예시로는 [동일한 테스트 이미지에서 두 모델을 비교한 테이블](https://wandb.ai/stacey/evalserver_answers_2/artifacts/results/eval_Daenerys/c2290abd3d7274f00ad8/files/eval_results.table.json#b6dae62d4f00d31eeebf$eval_Bob)을 참고해보세요.

### 모든 세부 정보 추적 및 전체 흐름 파악

특정 스텝의 예측값을 자세히 시각화하거나, 전체 집계 통계를 통해 에러 패턴과 개선 기회를 손쉽게 파악할 수 있습니다. 단일 모델 트레이닝 과정의 세부 구간을 비교하거나 여러 모델 버전의 결과를 함께 분석할 때 특히 유용합니다.

{{< img src="/images/data_vis/track_details.png" alt="실험 상세 추적" >}}

예시는 [MNIST 데이터셋에서 1회/5회 에포크 후 결과를 분석한 테이블](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json#7dd0cd845c0edb469dec)에서 확인할 수 있습니다.

## W&B Tables 활용 예시 Projects

아래는 실제 W&B Projects에서 W&B Tables를 어떻게 활용했는지 보여주는 사례들입니다.

### 이미지 분류

[이미지 분류 데이터 시각화](https://wandb.ai/stacey/mendeleev/reports/Visualize-Data-for-Image-Classification--VmlldzozNjE3NjA) 리포트를 읽거나, [데이터 시각화 Colab](https://wandb.me/dsviz-nature-colab) 튜토리얼을 따라해보고, [artifacts context](https://wandb.ai/stacey/mendeleev/artifacts/val_epoch_preds/val_pred_gawf9z8j/2dcee8fa22863317472b/files/val_epoch_res.table.json)를 확인해보세요. CNN이 [iNaturalist](https://www.inaturalist.org/pages/developers) 사진에서 10가지 생물(식물, 새, 곤충 등)을 어떻게 분류하는지 볼 수 있습니다.

{{< img src="/images/data_vis/image_classification.png" alt="두 모델 예측값에서 실제 라벨의 분포 비교" max-width="90%">}}

### 오디오

[Whale2Song - W&B Tables for Audio](https://wandb.ai/stacey/cshanty/reports/Whale2Song-W-B-Tables-for-Audio--Vmlldzo4NDI3NzM)에서 오디오 테이블을 직접 조작하면서 분석해 보세요. 고래 소리의 원본, 그리고 바이올린이나 트럼펫 소리 같은 악기로 합성된 버전까지 비교할 수 있습니다. 더불어 [오디오 트랜스퍼 Colab](http://wandb.me/audio-transfer)에서 직접 노래를 녹음하고 합성 결과까지 분석해 볼 수 있습니다.

{{< img src="/images/data_vis/audio.png" alt="오디오 테이블 예시" max-width="90%">}}

### 텍스트

트레이닝 데이터 또는 생성된 텍스트 샘플을 탐색하고, 관련 필드에 따라 동적으로 그룹화하여 다양한 모델 버전이나 실험 설정에 맞춰 쉽게 정렬·평가할 수 있습니다. 텍스트는 Markdown 형식으로 렌더링하거나, 시각적 차이 모드(diff mode)를 통해 비교할 수도 있습니다. [셰익스피어 텍스트 생성 리포트](https://wandb.ai/stacey/nlg/reports/Visualize-Text-Data-Predictions--Vmlldzo1NzcwNzY)에서 문자 기반 RNN 예시를 확인해보세요.

{{< img src="/images/data_vis/shakesamples.png" alt="hidden 레이어 크기 증가 시 더욱 창의적인 완성 예시" max-width="90%">}}

### 비디오

트레이닝 과정에서 기록된 비디오를 탐색하며 모델의 동작을 분석할 수 있습니다. 아래는 RL 에이전트가 [부작용 최소화](https://wandb.ai/stacey/saferlife/artifacts/video/videos_append-spawn/c1f92c6e27fa0725c154/files/video_examples.table.json)를 목표로 하는 [SafeLife benchmark](https://wandb.ai/safelife/v1dot2/benchmark)에서의 실험 기록 예시입니다.

{{< img src="/images/data_vis/video.png" alt="성공한 에이전트 탐색" max-width="90%">}}

### 표 형식 데이터

[표 형식 데이터 분할 및 전처리](https://wandb.ai/dpaiton/splitting-tabular-data/reports/Tabular-Data-Versioning-and-Deduplication-with-Weights-Biases--VmlldzoxNDIzOTA1) 리포트를 통해 버전 관리, 중복 제거 등 표 데이터 처리 방법을 참고하실 수 있습니다.

{{< img src="/images/data_vis/tabs.png" alt="Tables 및 Artifacts 워크플로우" max-width="90%">}}

### 모델 버전(시멘틱 세그멘테이션) 비교

[인터랙티브 노트북](https://wandb.me/dsviz-cars-demo)과 [라이브 예시](https://wandb.ai/stacey/evalserver_answers_2/artifacts/results/eval_Daenerys/c2290abd3d7274f00ad8/files/eval_results.table.json#a57f8e412329727038c2$eval_Ada)에서 시멘틱 세그멘테이션을 위한 Tables 로그와 모델 비교 사례를 볼 수 있습니다. [이 Table](https://wandb.ai/stacey/evalserver_answers_2/artifacts/results/eval_Daenerys/c2290abd3d7274f00ad8/files/eval_results.table.json)에서 직접 쿼리를 시도해보세요.

{{< img src="/images/data_vis/comparing_model_variants.png" alt="동일한 테스트 세트에서 두 모델의 최고 예측값 찾기" max-width="90%" >}}

### 트레이닝 시간에 따른 개선 분석

[예측값의 시간별 시각화](https://wandb.ai/stacey/mnist-viz/reports/Visualize-Predictions-over-Time--Vmlldzo1OTQxMTk)와 함께 연동되는 [인터랙티브 노트북](https://wandb.me/dsviz-mnist-colab)도 참고해보세요.