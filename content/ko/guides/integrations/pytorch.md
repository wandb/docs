---
title: PyTorch
menu:
  default:
    identifier: ko-guides-integrations-pytorch
    parent: integrations
weight: 300
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Intro_to_Weights_%26_Biases.ipynb" >}}

PyTorch는 Python에서 딥러닝을 위한 가장 인기 있는 프레임워크 중 하나이며, 특히 연구자들 사이에서 인기가 높습니다. W&B는 그래디언트 로깅부터 CPU 및 GPU에서 코드 프로파일링에 이르기까지 PyTorch를 위한 최고 수준의 지원을 제공합니다.

Colab 노트북에서 인테그레이션을 사용해 보세요.

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Simple_PyTorch_Integration.ipynb" >}}

[예제 저장소](https://github.com/wandb/examples)에서 스크립트를 확인할 수도 있습니다. 여기에는 [Fashion MNIST](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cnn-fashion)에서 [Hyperband](https://arxiv.org/abs/1603.06560)를 사용하는 하이퍼파라미터 최적화와 이를 통해 생성되는 [W&B Dashboard](https://wandb.ai/wandb/keras-fashion-mnist/runs/5z1d85qs)가 포함됩니다.

## `wandb.watch`로 그래디언트 로그하기

그래디언트를 자동으로 기록하려면 [`wandb.watch`]({{< relref path="/ref/python/watch.md" lang="ko" >}})를 호출하고 PyTorch 모델을 전달하면 됩니다.

```python
import wandb

wandb.init(config=args)

model = ...  # 모델 설정

# Magic
wandb.watch(model, log_freq=100)

model.train()
for batch_idx, (data, target) in enumerate(train_loader):
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % args.log_interval == 0:
        wandb.log({"loss": loss})
```

동일한 스크립트에서 여러 모델을 추적해야 하는 경우 각 모델에서 개별적으로 `wandb.watch`를 호출할 수 있습니다. 이 함수에 대한 참조 문서는 [여기]({{< relref path="/ref/python/watch.md" lang="ko" >}})에 있습니다.

{{% alert color="secondary" %}}
그래디언트, 메트릭 및 그래프는 순방향 _및_ 역방향 패스 후에 `wandb.log`가 호출될 때까지 기록되지 않습니다.
{{% /alert %}}

## 이미지 및 미디어 로그

이미지 데이터와 함께 PyTorch `Tensors`를 [`wandb.Image`]({{< relref path="/ref/python/data-types/image.md" lang="ko" >}})로 전달하면 [`torchvision`](https://pytorch.org/vision/stable/index.html)의 유틸리티가 자동으로 이미지를 변환하는 데 사용됩니다.

```python
images_t = ...  # PyTorch 텐서로 이미지 생성 또는 로드
wandb.log({"examples": [wandb.Image(im) for im in images_t]})
```

PyTorch 및 기타 프레임워크에서 W&B에 풍부한 미디어를 로깅하는 방법에 대한 자세한 내용은 [미디어 로깅 가이드]({{< relref path="/guides/models/track/log/media.md" lang="ko" >}})를 확인하세요.

모델의 예측값 또는 파생된 메트릭과 같은 정보를 미디어와 함께 포함하려면 `wandb.Table`을 사용하세요.

```python
my_table = wandb.Table()

my_table.add_column("image", images_t)
my_table.add_column("label", labels)
my_table.add_column("class_prediction", predictions_t)

# W&B에 테이블 로그
wandb.log({"mnist_predictions": my_table})
```

{{< img src="/images/integrations/pytorch_example_table.png" alt="The code above generates a table like this one. This model's looking good!" >}}

데이터셋과 Models 로깅 및 시각화에 대한 자세한 내용은 [W&B Tables 가이드]({{< relref path="/guides/core/tables/" lang="ko" >}})를 확인하세요.

## PyTorch 코드 프로파일링

{{< img src="/images/integrations/pytorch_example_dashboard.png" alt="View detailed traces of PyTorch code execution inside W&B dashboards." >}}

W&B는 [PyTorch Kineto](https://github.com/pytorch/kineto)의 [Tensorboard 플러그인](https://github.com/pytorch/kineto/blob/master/tb_plugin/README.md)과 직접 통합되어 PyTorch 코드 프로파일링, CPU 및 GPU 통신 세부 정보 검사, 병목 현상 및 최적화 식별을 위한 툴을 제공합니다.

```python
profile_dir = "path/to/run/tbprofile/"
profiler = torch.profiler.profile(
    schedule=schedule,  # 스케줄링에 대한 자세한 내용은 프로파일러 문서 참조
    on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir),
    with_stack=True,
)

with profiler:
    ...  # 프로파일링할 코드를 여기에 실행
    # 자세한 사용법은 프로파일러 문서 참조

# wandb Artifact 생성
profile_art = wandb.Artifact("trace", type="profile")
# pt.trace.json 파일을 Artifact에 추가
profile_art.add_file(glob.glob(profile_dir + ".pt.trace.json"))
# 아티팩트 로그
profile_art.save()
```

[이 Colab](http://wandb.me/trace-colab)에서 작동하는 예제 코드를 확인하고 실행하세요.

{{% alert color="secondary" %}}
대화형 추적 보기 툴은 Chrome 브라우저에서 가장 잘 작동하는 Chrome Trace Viewer를 기반으로 합니다.
{{% /alert %}}
