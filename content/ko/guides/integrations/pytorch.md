---
title: PyTorch
menu:
  default:
    identifier: ko-guides-integrations-pytorch
    parent: integrations
weight: 300
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Intro_to_Weights_%26_Biases.ipynb" >}}

PyTorch 는 특히 연구자들 사이에서 파이썬 딥러닝에 가장 널리 사용되는 프레임워크 중 하나입니다. W&B 는 PyTorch 의 그레이디언트 로깅부터 CPU 및 GPU 환경에서의 코드 프로파일링까지 완전하게 지원합니다.

Colab 노트북에서 W&B PyTorch 인테그레이션을 직접 체험해 보세요.

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Simple_PyTorch_Integration.ipynb" >}}

또한, [example repo](https://github.com/wandb/examples)에서 다양한 스크립트를 확인할 수 있고, [Hyperband](https://arxiv.org/abs/1603.06560)를 활용한 하이퍼파라미터 최적화 예제, [Fashion MNIST](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cnn-fashion) 등 대표 예시, 그리고 이로부터 생성되는 [W&B Dashboard](https://wandb.ai/wandb/keras-fashion-mnist/runs/5z1d85qs) 도 볼 수 있습니다.

## `run.watch`로 그레이디언트 로깅하기

그레이디언트를 자동으로 로깅하려면, [`wandb.Run.watch()`]({{< relref path="/ref/python/sdk/classes/run.md/#method-runwatch" lang="ko" >}})를 호출하면서 PyTorch 모델을 인자로 넘겨주면 됩니다.

```python
import wandb

with wandb.init(config=args) as run:

    model = ...  # 모델을 준비합니다

    # 매직!
    run.watch(model, log_freq=100)

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            run.log({"loss": loss})
```

하나의 스크립트에서 여러 모델을 추적하고 싶다면, 각각의 모델에 대해 [`wandb.Run.watch()`]({{< relref path="/ref/python/sdk/classes/run/#method-runwatch" lang="ko" >}})를 따로 호출해주면 됩니다.

{{% alert color="secondary" %}}
그레이디언트, 메트릭, 그래프는 _forward_ 와 _backward_ 패스가 모두 끝나고 `wandb.Run.log()`가 호출되기 전까지는 로깅되지 않습니다.
{{% /alert %}}

## 이미지 및 미디어 로깅

PyTorch 의 `Tensor`로 만들어진 이미지를 [`wandb.Image`]({{< relref path="/ref/python/sdk/data-types/image.md" lang="ko" >}})에 전달하면, [`torchvision`](https://pytorch.org/vision/stable/index.html) 유틸리티와 함께 자동으로 변환할 수 있습니다.

```python
with wandb.init(project="my_project", entity="my_entity") as run:
    images_t = ...  # 이미지를 PyTorch Tensor로 생성하거나 불러옵니다
    run.log({"examples": [wandb.Image(im) for im in images_t]})
```

PyTorch 및 기타 프레임워크에서 다양한 미디어 데이터를 로깅하는 방법은 [미디어 로깅 가이드]({{< relref path="/guides/models/track/log/media.md" lang="ko" >}})를 참고하세요.

이미지 등 미디어와 함께 모델 예측값이나 계산된 메트릭을 함께 저장하고 싶다면 `wandb.Table`을 활용할 수 있습니다.

```python
with wandb.init() as run:
    my_table = wandb.Table()

    my_table.add_column("image", images_t)
    my_table.add_column("label", labels)
    my_table.add_column("class_prediction", predictions_t)

    # Table 을 W&B에 로그
    run.log({"mnist_predictions": my_table})
```

{{< img src="/images/integrations/pytorch_example_table.png" alt="PyTorch 모델 결과" >}}

데이터셋 및 모델을 로깅하고 시각화하는 다양한 방법은 [W&B Tables 가이드]({{< relref path="/guides/models/tables/" lang="ko" >}})에서 확인할 수 있습니다.

## PyTorch 코드 프로파일링

{{< img src="/images/integrations/pytorch_example_dashboard.png" alt="PyTorch 실행 트레이스" >}}

W&B 는 [PyTorch Kineto](https://github.com/pytorch/kineto)의 [Tensorboard plugin](https://github.com/pytorch/kineto/blob/master/tb_plugin/README.md)과 연동하여, PyTorch 코드 프로파일링과 CPU, GPU 간 통신 디테일 파악, 병목 구간 탐지 및 최적화 가능성 진단을 위한 다양한 도구를 지원합니다.

```python
profile_dir = "path/to/run/tbprofile/"
profiler = torch.profiler.profile(
    schedule=schedule,  # 스케줄에 대한 자세한 설명은 profiler 공식 문서를 참고하세요
    on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir),
    with_stack=True,
)

with profiler:
    ...  # 여기서 프로파일링할 코드를 실행하세요
    # 자세한 내용은 profiler 공식 문서를 참고하세요

# wandb Artifact 생성
profile_art = wandb.Artifact("trace", type="profile")
# pt.trace.json 파일을 Artifact에 추가
profile_art.add_file(glob.glob(profile_dir + ".pt.trace.json"))
# 아티팩트 로그 저장
profile_art.save()
```

동작 예시는 [이 Colab](https://wandb.me/trace-colab)에서 확인할 수 있습니다.

{{% alert color="secondary" %}}
인터랙티브 트레이스 뷰어는 크롬의 트레이스 뷰어를 기반으로 하며, 크롬 브라우저에서 가장 잘 작동합니다.
{{% /alert %}}