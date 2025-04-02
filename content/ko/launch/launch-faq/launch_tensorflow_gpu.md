---
title: How do I make W&B Launch work with Tensorflow on GPU?
menu:
  launch:
    identifier: ko-launch-launch-faq-launch_tensorflow_gpu
    parent: launch-faq
---

GPU를 사용하는 TensorFlow 작업의 경우, 컨테이너 빌드를 위한 사용자 정의 기본 이미지를 지정하세요. 이렇게 하면 run 동안 적절한 GPU 활용이 보장됩니다. 리소스 설정에서 `builder.accelerator.base_image` 키 아래에 이미지 태그를 추가합니다. 예시:

```json
{
    "gpus": "all",
    "builder": {
        "accelerator": {
            "base_image": "tensorflow/tensorflow:latest-gpu"
        }
    }
}
```

W&B 0.15.6 이전 버전에서는 `base_image`의 상위 키로 `accelerator` 대신 `cuda`를 사용하세요.
