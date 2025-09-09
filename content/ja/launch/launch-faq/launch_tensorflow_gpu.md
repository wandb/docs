---
title: W&B ローンンチ を TensorFlow で GPU 上で動作させるにはどうすればよいですか？
menu:
  launch:
    identifier: ja-launch-launch-faq-launch_tensorflow_gpu
    parent: launch-faq
---

GPU を使用する TensorFlow ジョブでは、コンテナのビルドに使用するカスタムのベースイメージを指定します。これにより Runs 中に GPU を適切に活用できます。リソース設定の `builder.accelerator.base_image` キーの下に image タグを追加します。例えば:

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

W&B 0.15.6 より前のバージョンでは、`base_image` の親キーには `accelerator` ではなく `cuda` を使用してください。