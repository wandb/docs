---
title: 가속기 베이스 이미지에 필요한 요구 사항은 무엇인가요?
menu:
  launch:
    identifier: ko-launch-launch-faq-requirements_accelerator_base_image
    parent: launch-faq
---

가속기를 사용하는 작업의 경우, 필요한 가속기 구성 요소가 포함된 기본 이미지를 제공해야 합니다. 가속기 이미지에 대해 다음 요구 사항을 충족해야 합니다.

- Debian과의 호환성 (Launch Dockerfile은 apt-get을 사용해 Python을 설치합니다)
- 지원되는 CPU 및 GPU 하드웨어 명령어 세트 (의도한 GPU 와 CUDA 버전 호환성 확인)
- 제공된 가속기 버전과 기계학습 알고리즘에서 사용하는 패키지 간의 호환성
- 하드웨어 호환성을 위해 추가 설치가 필요한 패키지의 사전 설치