---
title: W&B Launch 는 이미지를 어떻게 빌드하나요?
menu:
  launch:
    identifier: ko-launch-launch-faq-launch_build_images
    parent: launch-faq
---

이미지 빌드 단계는 job 소스와 리소스 설정에서 지정한 accelerator base 이미지에 따라 달라집니다.

{{% alert %}}
큐를 설정하거나 job 을 제출할 때, 반드시 큐 또는 job 리소스 설정에 base accelerator 이미지를 포함해야 합니다:
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

빌드 프로세스에는 job 유형과 제공된 accelerator base 이미지에 따라 다음과 같은 작업이 포함됩니다:

| | apt 를 사용해 Python 설치 | Python 패키지 설치 | 사용자 및 작업 디렉토리 생성 | 코드를 이미지에 복사 | entrypoint 설정 | |