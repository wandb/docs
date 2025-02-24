---
title: What requirements does the accelerator base image have?
menu:
  launch:
    identifier: ko-launch-launch-faq-requirements_accelerator_base_image
    parent: launch-faq
---

가속기를 활용하는 작업의 경우 필요한 가속기 구성 요소가 포함된 기본 이미지를 제공하십시오. 가속기 이미지에 대한 다음 요구 사항을 확인하십시오.

- Debian 과의 호환성 (Launch Dockerfile 은 apt-get 을 사용하여 Python 을 설치함)
- 지원되는 CPU 및 GPU 하드웨어 명령어 집합 (의도한 GPU 와 CUDA 버전 호환성을 확인)
- 제공된 가속기 버전과 기계학습 알고리즘의 패키지 간의 호환성
- 하드웨어 호환성을 위해 추가 단계가 필요한 패키지 설치