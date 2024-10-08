---
title: PyTorch
displayed_sidebar: default
---
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

<CTAButtons colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Intro_to_Weights_%26_Biases.ipynb"></CTAButtons>

PyTorch는 Python에서 딥러닝을 위한 가장 인기 있는 프레임워크 중 하나로, 특히 연구자들 사이에서 인기가 많습니다. W&B는 PyTorch에 대한 일류 지원을 제공하여 그레이디언트 로그부터 CPU 및 GPU에서의 코드 프로파일링까지 가능합니다.

:::info
Colab 노트북에서 우리의 인테그레이션을 시도해보세요.

<CTAButtons colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Simple_PyTorch_Integration.ipynb"></CTAButtons>

[Hyperband](https://arxiv.org/abs/1603.06560)를 사용한 하이퍼파라미터 최적화를 포함하여 스크립트의 경우 [예제 저장소](https://github.com/wandb/examples)를 확인하십시오. 이 저장소에는 [Fashion MNIST](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cnn-fashion)에 대한 예제가 포함되어 있으며, 생성된 [W&B 대시보드](https://wandb.ai/wandb/keras-fashion-mnist/runs/5z1d85qs)도 볼 수 있습니다.
:::

## `wandb.watch`로 그레이디언트 로그하기

그레이디언트를 자동으로 로그하려면, [`wandb.watch`](../../ref/python/watch.md)를 호출하고 PyTorch 모델을 전달할 수 있습니다.

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

같은 스크립트에서 여러 모델을 추적해야 한다면, 각 모델에 대해 별도로 `wandb.watch`를 호출할 수 있습니다. 이 함수에 대한 참고 문서는 [여기](../../ref/python/watch.md)에 있습니다.

:::caution
`wandb.log`이 순전파 _그리고_ 역전파 후에 호출되기 전까지 그레이디언트, 메트릭, 그래프는 로그되지 않습니다.
:::

## 이미지 및 미디어 로그하기

PyTorch `Tensors`에 이미지 데이터를 담아 [`wandb.Image`](../../ref/python/data-types/image.md)에 전달하면, [`torchvision`](https://pytorch.org/vision/stable/index.html) 유틸리티가 이를 자동으로 이미지로 변환합니다:

```python
images_t = ...  # PyTorch Tensor로 이미지 생성 또는 로드
wandb.log({"examples": [wandb.Image(im) for im in images_t]})
```

PyTorch 및 다른 프레임워크에서의 풍부한 미디어 로그에 대한 자세한 내용은 우리의 [미디어 로그 가이드](../track/log/media.md)를 확인하십시오.

미디어와 함께 모델의 **예측값**이나 도출된 메트릭과 같은 정보를 포함하려면 `wandb.Table`을 사용하십시오.

```python
my_table = wandb.Table()

my_table.add_column("image", images_t)
my_table.add_column("label", labels)
my_table.add_column("class_prediction", predictions_t)

# Table을 W&B에 로그
wandb.log({"mnist_predictions": my_table})
```

![위 코드는 이와 같은 테이블을 생성합니다. 이 모델은 상태가 좋습니다!](/images/integrations/pytorch_example_table.png)

데이터셋과 모델을 로그하고 시각화하는 것에 대한 더 많은 정보를 얻으려면, 우리의 [W&B Tables 가이드](../tables/intro.md)를 확인하십시오.

## PyTorch 코드 프로파일링

![W&B 대시보드에서 PyTorch 코드 실행의 상세 추적을 확인하세요.](/images/integrations/pytorch_example_dashboard.png)

W&B는 [PyTorch Kineto](https://github.com/pytorch/kineto)의 [Tensorboard 플러그인](https://github.com/pytorch/kineto/blob/master/tb_plugin/README.md)과 직접 통합되어, PyTorch 코드 프로파일링, CPU 및 GPU 통신의 세부 사항 검사, 병목 현상 및 최적화 식별을 위한 도구를 제공합니다.

```python
profile_dir = "path/to/run/tbprofile/"
profiler = torch.profiler.profile(
    schedule=schedule,  # 일정에 대한 자세한 정보는 프로파일러 문서를 참고하세요
    on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir),
    with_stack=True,
)

with profiler:
    ...  # 프로파일링할 코드를 여기에 실행하세요
    # 자세한 사용 정보는 프로파일러 문서를 참고하세요

# wandb 아티팩트 생성
profile_art = wandb.Artifact("trace", type="profile")
# pt.trace.json 파일을 아티팩트에 추가
profile_art.add_file(glob.glob(profile_dir + ".pt.trace.json"))
# 아티팩트를 로그
profile_art.save()
```

[이 Colab](http://wandb.me/trace-colab)에서 작동 중인 예제 코드를 확인하고 실행해보세요.

:::caution
인터랙티브 추적 뷰잉 툴은 Chrome Trace Viewer를 기반으로 하며, Chrome 브라우저에서 최적의 작동을 합니다.
:::