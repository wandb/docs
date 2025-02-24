---
title: How does W&B Launch build images?
menu:
  launch:
    identifier: ko-launch-launch-faq-launch_build_images
    parent: launch-faq
---

이미지 빌드 단계는 job 소스와 리소스 설정에 지정된 accelerator 기본 이미지에 따라 달라집니다.

{{% alert %}}
대기열을 설정하거나 job을 제출할 때 대기열 또는 job 리소스 설정에 기본 accelerator 이미지를 포함하세요.
```json
{
    "builder": {
        "accelerator": {
            "base_image": "image-name"
        }
    }
}
```
{{% /alert %}}

빌드 프로세스는 job 유형 및 제공된 accelerator 기본 이미지를 기반으로 다음 작업을 포함합니다.

| | apt를 사용하여 Python 설치 | Python 패키지 설치 | 사용자 및 작업 디렉터리 생성 | 코드를 이미지에 복사 | 진입점 설정 | |
