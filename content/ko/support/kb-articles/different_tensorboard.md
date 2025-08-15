---
title: W&B는 TensorBoard와 어떻게 다른가요?
menu:
  support:
    identifier: ko-support-kb-articles-different_tensorboard
support:
- 텐서보드
toc_hide: true
type: docs
url: /support/:filename
---

W&B는 TensorBoard와 연동되며 experiment tracking 툴을 향상시킵니다. 창립자들은 TensorBoard 사용자들이 자주 겪는 여러 불편함을 해결하기 위해 W&B를 만들었습니다. 주요 개선점은 다음과 같습니다:

1. **Model Reproducibility**: W&B는 실험, 탐색, 그리고 모델 재현성을 쉽게 만들어줍니다. 메트릭, 하이퍼파라미터, 코드 버전, 그리고 모델 체크포인트를 모두 기록해 재현성을 보장합니다.

2. **Automatic Organization**: W&B는 모든 시도한 모델에 대한 개요를 제공해 프로젝트 인수인계나 휴가 중에도 효율적으로 관리할 수 있게 해줍니다. 덕분에 옛 실험의 반복 실행을 방지해 시간을 절약할 수 있습니다.

3. **Quick Integration**: W&B를 프로젝트에 5분 만에 통합할 수 있습니다. 오픈소스 파이썬 패키지를 설치하고 코드 몇 줄만 추가하세요. 기록된 메트릭과 정보는 각 모델 run에서 자동으로 확인할 수 있습니다.

4. **Centralized Dashboard**: 트레이닝이 어디서 이루어져도—로컬, 연구실 클러스터, 클라우드 spot 인스턴스 등—일관된 대시보드를 엑세스할 수 있습니다. 다양한 머신에서 TensorBoard 파일을 따로 관리할 필요가 없습니다.

5. **Robust Filtering Table**: 다양한 모델의 결과를 편리하게 검색, 필터, 정렬, 그룹화할 수 있습니다. 여러 작업에서 가장 성능이 좋은 모델도 쉽게 찾을 수 있습니다. 이 부분은 대규모 프로젝트에서 TensorBoard가 종종 한계를 보이는 부분입니다.

6. **Collaboration Tools**: W&B는 복잡한 기계학습 프로젝트에서 협업을 강화합니다. 프로젝트 링크를 공유하고, 프라이빗 팀을 활용하여 결과를 안전하게 공유하세요. 대화형 시각화와 마크다운 설명으로 리포트를 만들어 작업 로그나 발표 자료로도 활용할 수 있습니다.