---
title: 예시 Reports
description: Reports 갤러리
menu:
  default:
    identifier: ko-guides-core-reports-reports-gallery
    parent: reports
weight: 70
---

## 노트: 빠른 요약과 함께 시각화 추가하기

프로젝트 개발 중에 발견한 중요한 관찰 결과, 향후 작업 아이디어, 혹은 달성한 이정표를 기록하세요. 리포트에 포함된 모든 experiment run 들은 각자의 파라미터, 메트릭, 로그, 코드에 링크되어 있어 작업 전체의 맥락을 보존할 수 있습니다.

간단한 텍스트를 메모해두고, 인사이트를 보여줄 수 있는 관련 차트들을 추가해보세요.

트레이닝 시간이 너무 느린 경우를 비교해 공유하는 예시는 [What To Do When Inception-ResNet-V2 Is Too Slow](https://wandb.ai/stacey/estuary/reports/When-Inception-ResNet-V2-is-too-slow--Vmlldzo3MDcxMA) W&B Report에서 확인할 수 있습니다.

{{< img src="/images/reports/notes_add_quick_summary.png" alt="빠른 요약 노트" max-width="90%">}}

복잡한 코드베이스에서 나온 최고의 예시들을 저장하여 나중에 참고하거나 상호작용하기 편리하게 만드세요. Lyft 데이터셋의 LIDAR 포인트 클라우드를 시각화하고 3D 바운딩 박스로 주석을 추가하는 예시는 [LIDAR point clouds](https://wandb.ai/stacey/lyft/reports/LIDAR-Point-Clouds-of-Driving-Scenes--Vmlldzo2MzA5Mg) W&B Report에서 볼 수 있습니다.

{{< img src="/images/reports/notes_add_quick_summary_save_best_examples.png" alt="최고의 예시 저장" max-width="90%" >}}

## 협업: 동료와 발견한 내용을 공유하세요

프로젝트 시작 방법을 설명하고, 지금까지 관찰한 점을 공유하며, 최신 발견한 내용을 정리해 보세요. 동료들은 패널이나 리포트 맨 끝에 댓글로 제안이나 상세 토의를 남길 수 있습니다.

동적으로 설정을 바꿀 수 있도록 포함해 두면, 동료들이 직접 탐색해 추가적인 인사이트를 얻고, 다음 단계도 더 잘 계획할 수 있습니다. 이 예시에서는 세 가지 유형의 experiment 를 각각 시각화하거나, 비교하거나, 평균을 낼 수 있습니다.

벤치마크의 첫 run, 관찰 결과 공유 방법은 [SafeLife benchmark experiments](https://wandb.ai/stacey/saferlife/reports/SafeLife-Benchmark-Experiments--Vmlldzo0NjE4MzM) W&B Report에서 참고할 수 있습니다.

{{< img src="/images/reports/intro_collaborate1.png" alt="SafeLife 벤치마크 리포트" >}}

{{< img src="/images/reports/intro_collaborate2.png" alt="Experiment 비교 화면" >}}

슬라이더와 설정 가능한 미디어 패널로 모델 결과나 트레이닝 진행 상황을 보여줄 수 있습니다. 슬라이더가 포함된 W&B Report의 좋은 예시는 [Cute Animals and Post-Modern Style Transfer: StarGAN v2 for Multi-Domain Image Synthesis](https://wandb.ai/stacey/stargan/reports/Cute-Animals-and-Post-Modern-Style-Transfer-StarGAN-v2-for-Multi-Domain-Image-Synthesis---VmlldzoxNzcwODQ) 리포트에서 볼 수 있습니다.

{{< img src="/images/reports/intro_collaborate3.png" alt="슬라이더가 있는 StarGAN 리포트" >}}

{{< img src="/images/reports/intro_collaborate4.png" alt="인터랙티브 미디어 패널" >}}

## 작업 로그: 시도해 본 것과 다음 단계를 기록하세요

실험에 대한 생각, 발견한 내용, 시행착오, 그리고 다음 단계 계획을 프로젝트 진행 중에 적어두고, 모든 것을 한 곳에 정리하세요. 이 과정에서 스크립트 외에도 중요한 내용들을 모두 '문서화'할 수 있습니다. 발견한 내용을 어떻게 리포트할 수 있는지는 [Who Is Them? Text Disambiguation With Transformers](https://wandb.ai/stacey/winograd/reports/Who-is-Them-Text-Disambiguation-with-Transformers--VmlldzoxMDU1NTc) W&B Report에서 예시를 볼 수 있습니다.

{{< img src="/images/reports/intro_work_log_1.png" alt="텍스트 의미 구별 리포트" >}}

프로젝트의 스토리를 기록해 두면, 나중에 본인이나 다른 사람들이 모델이 왜, 어떻게 개발됐는지 이해할 수 있습니다. 관련 예시는 [The View from the Driver's Seat](https://wandb.ai/stacey/deep-drive/reports/The-View-from-the-Driver-s-Seat--Vmlldzo1MTg5NQ) W&B Report에서 확인하실 수 있습니다.

{{< img src="/images/reports/intro_work_log_2.png" alt="운전석 프로젝트 리포트" >}}

W&B Reports가 사용된 실제 대형 기계학습 프로젝트 사례를 알아보고 싶다면, [Learning Dexterity End-to-End Using W&B Reports](https://bit.ly/wandb-learning-dexterity)를 참고해보세요. OpenAI Robotics 팀이 W&B Reports로 대규모 machine learning 프로젝트를 어떻게 진행했는지 볼 수 있습니다.