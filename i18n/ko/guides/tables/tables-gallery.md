---
description: Examples of W&B Tables
displayed_sidebar: default
---

# 테이블 갤러리
다음 섹션들은 테이블을 사용할 수 있는 몇 가지 방법을 강조합니다:

### 데이터 보기

모델 학습 또는 평가 중에 메트릭과 풍부한 미디어를 기록한 다음, 클라우드 또는 [호스팅 인스턴스](https://docs.wandb.ai/guides/hosting)에 동기화된 지속적인 데이터베이스에서 결과를 시각화합니다.

![예제를 탐색하고 데이터의 수량과 분포를 확인합니다](/images/data_vis/tables_see_data.png)

예를 들어, [균형 잡힌 사진 데이터세트의 분할을 보여주는 이 테이블](https://wandb.ai/stacey/mendeleev/artifacts/balanced\_data/inat\_80-10-10\_5K/ab79f01e007113280018/files/data\_split.table.json)을 확인하세요.

### 데이터를 상호작용적으로 탐색하기

테이블을 보고, 정렬하고, 필터링하고, 그룹화하고, 조인하고, 쿼리하여 데이터와 모델 성능을 이해합니다. 정적 파일을 탐색하거나 분석 스크립트를 다시 실행할 필요가 없습니다.

![원본 노래와 그것의 합성 버전(탬버 전송 포함)을 듣습니다](/images/data_vis/explore_data.png)

예를 들어, [스타일 전송된 오디오에 대한 이 리포트](https://wandb.ai/stacey/cshanty/reports/Whale2Song-W-B-Tables-for-Audio--Vmlldzo4NDI3NzM)를 보세요.

### 모델 버전 비교

다른 학습 에포크, 데이터세트, 하이퍼파라미터 선택, 모델 아키텍처 등의 결과를 빠르게 비교합니다.

![세밀한 차이를 확인: 왼쪽 모델은 일부 빨간색 인도를 감지하지만, 오른쪽 모델은 그렇지 않습니다.](/images/data_vis/compare_model_versions.png)

예를 들어, [동일한 테스트 이미지에 대한 두 모델을 비교하는 이 테이블](https://wandb.ai/stacey/evalserver\_answers\_2/artifacts/results/eval\_Daenerys/c2290abd3d7274f00ad8/files/eval\_results.table.json#b6dae62d4f00d31eeebf$eval\_Bob)을 보세요.

### 모든 세부 사항을 추적하고 큰 그림 보기

특정 예측값을 특정 단계에서 시각화하여 확대합니다. 오류 패턴을 식별하고 개선 기회를 이해하려면 집계 통계로 확대 축소합니다. 이 도구는 단일 모델 학습의 단계를 비교하거나 다른 모델 버전의 결과를 비교하는 데 사용할 수 있습니다.

![](/images/data_vis/track_details.png)

예를 들어, [MNIST 데이터세트에서 한 에포크 후 그리고 다섯 에포크 후의 결과를 분석하는 예제 테이블](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json#7dd0cd845c0edb469dec)을 보세요.

## W&B 테이블을 사용한 예제 프로젝트
다음은 W&B 테이블을 사용하는 실제 W&B 프로젝트를 강조합니다.

### 이미지 분류

[이 리포트](https://wandb.ai/stacey/mendeleev/reports/Visualize-Data-for-Image-Classification--VmlldzozNjE3NjA)를 읽거나, [이 코랩](https://wandb.me/dsviz-nature-colab)을 따라하거나, [iNaturalist](https://www.inaturalist.org/pages/developers) 사진에서 식물, 새, 곤충 등 생물 10종을 식별하는 CNN을 보여주는 [아티팩트 컨텍스트](https://wandb.ai/stacey/mendeleev/artifacts/val\_epoch\_preds/val\_pred\_gawf9z8j/2dcee8fa22863317472b/files/val\_epoch\_res.table.json)를 탐색하세요.

![두 다른 모델의 예측에 걸친 진짜 레이블의 분포를 비교합니다.](/images/data_vis/image_classification.png)

### 오디오

탬버 전송에 대한 [이 리포트](https://wandb.ai/stacey/cshanty/reports/Whale2Song-W-B-Tables-for-Audio--Vmlldzo4NDI3NzM)에서 오디오 테이블을 상호 작용합니다. 같은 멜로디의 녹음된 고래 노래와 바이올린이나 트럼펫과 같은 악기에서 합성된 버전을 비교할 수 있습니다. 또한 [이 코랩](http://wandb.me/audio-transfer)을 사용하여 자신의 노래를 녹음하고 W&B에서 합성된 버전을 탐색할 수 있습니다.

![](/images/data_vis/audio.png)

### 텍스트

학습 데이터나 생성된 출력에서 텍스트 샘플을 탐색하고, 관련 필드별로 동적으로 그룹화하고, 모델 변형이나 실험 설정에 걸쳐 평가를 조정합니다. Markdown으로 텍스트를 렌더링하거나 텍스트를 비교하기 위해 시각적 차이 모드를 사용하세요. [이 리포트](https://wandb.ai/stacey/nlg/reports/Visualize-Text-Data-Predictions--Vmlldzo1NzcwNzY)에서 셰익스피어를 생성하기 위한 간단한 문자 기반 RNN을 탐색하세요.

![은닉층의 크기를 두 배로 늘리면 더 창의적인 프롬프트 완성을 얻을 수 있습니다.](@site/static/images/data_vis/shakesamples.png)

### 비디오

학습 중에 기록된 비디오를 탐색하고 집계하여 모델을 이해합니다. 여기에는 RL 에이전트가 [부작용을 최소화](https://wandb.ai/stacey/saferlife/artifacts/video/videos\_append-spawn/c1f92c6e27fa0725c154/files/video\_examples.table.json)하려는 [SafeLife 벤치마크](https://wandb.ai/safelife/v1dot2/benchmark)를 사용하는 초기 예가 있습니다.

![몇 안 되는 성공적인 에이전트를 쉽게 탐색합니다](/images/data_vis/video.png)

### 표 데이터

버전 관리와 중복 제거를 사용하여 [표 데이터를 분할하고 전처리하는 방법](https://wandb.ai/dpaiton/splitting-tabular-data/reports/Tabular-Data-Versioning-and-Deduplication-with-Weights-Biases--VmlldzoxNDIzOTA1)에 대한 리포트를 봅니다.

![테이블과 아티팩트가 함께 작동하여 데이터셋 반복을 버전 관리, 라벨링, 중복 제거합니다](@site/static/images/data_vis/tabs.png)

### 모델 변형 비교 (semantic segmentation)

semantic segmentation에 대한 테이블을 기록하고 다른 모델을 비교하는 [인터랙티브 노트북](https://wandb.me/dsviz-cars-demo)과 [실시간 예제](https://wandb.ai/stacey/evalserver\_answers\_2/artifacts/results/eval\_Daenerys/c2290abd3d7274f00ad8/files/eval\_results.table.json#a57f8e412329727038c2$eval\_Ada)입니다. [이 테이블](https://wandb.ai/stacey/evalserver\_answers\_2/artifacts/results/eval\_Daenerys/c2290abd3d7274f00ad8/files/eval\_results.table.json)에서 자신의 쿼리를 시도해 보세요.

![동일한 테스트 세트에서 두 모델의 최고 예측값을 찾습니다](/images/data_vis/comparing_model_variants.png)

### 학습 시간에 따른 개선 분석

시간에 따른 예측값을 [시각화하는 방법](https://wandb.ai/stacey/mnist-viz/reports/Visualize-Predictions-over-Time--Vmlldzo1OTQxMTk)에 대한 자세한 리포트와 해당 [인터랙티브 노트북](https://wandb.me/dsviz-mnist-colab).