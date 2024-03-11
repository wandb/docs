---
displayed_sidebar: default
---

# PyTorch

[**여기에서 Colab 노트북으로 시도해보세요 →**](http://wandb.me/intro)

PyTorch는 연구원들 사이에서 특히 인기가 많은 Python의 딥러닝 프레임워크 중 하나입니다. W&B는 그레이디언트 로깅부터 CPU와 GPU에서 코드 프로파일링까지 PyTorch에 대한 일류 지원을 제공합니다.

:::info
[colab 노트북](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Simple\_PyTorch\_Integration.ipynb)에서 우리의 인테그레이션을 시도해보세요(아래에 동영상 튜토리얼 포함) 또는 [Fashion MNIST](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cnn-fashion)에서 [Hyperband](https://arxiv.org/abs/1603.06560)를 사용한 하이퍼파라미터 최적화를 포함한 스크립트, 그리고 이것이 생성하는 [W&B 대시보드](https://wandb.ai/wandb/keras-fashion-mnist/runs/5z1d85qs)에 대한 우리의 [예제 저장소](https://github.com/wandb/examples)를 확인해 보세요.
:::

## `wandb.watch`로 그레이디언트 로깅하기

자동으로 그레이디언트를 로깅하려면, PyTorch 모델을 전달하면서 [`wandb.watch`](../../ref/python/watch.md)를 호출할 수 있습니다.

```python
import wandb

wandb.init(config=args)

model = ...  # 모델 설정

# 마법
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

동일한 스크립트에서 여러 모델을 추적해야 하는 경우, 각 모델에 대해 `wandb.watch`를 별도로 호출할 수 있습니다. 이 함수에 대한 참조 문서는 [여기](../../ref/python/watch.md)에 있습니다.

:::caution
그레이디언트, 메트릭 및 그래프는 전방 통과 _및_ 역방향 통과 후 `wandb.log`가 호출될 때까지 로깅되지 않습니다.
:::

## 이미지와 미디어 로깅하기

PyTorch `Tensors`에 이미지 데이터를 [`wandb.Image`](../../ref/python/data-types/image.md)로 전달하면, [`torchvision`](https://pytorch.org/vision/stable/index.html)의 유틸리티가 자동으로 이미지로 변환되어 사용됩니다:

```python
images_t = ...  # PyTorch Tensors로 이미지 생성 또는 로드
wandb.log({"examples": [wandb.Image(im) for im in images_t]})
```

PyTorch 및 기타 프레임워크에서 W&B로 리치 미디어를 로깅하는 방법에 대한 자세한 내용은 [미디어 로깅 가이드](../track/log/media.md)를 확인하세요.

모델의 예측값이나 파생 메트릭과 같이 미디어와 함께 정보를 포함하고 싶다면, `wandb.Table`을 사용하세요.

```python
my_table = wandb.Table()

my_table.add_column("image", images_t)
my_table.add_column("label", labels)
my_table.add_column("class_prediction", predictions_t)

# W&B에 Table 로깅하기
wandb.log({"mnist_predictions": my_table})
```

![위 코드는 이와 같은 테이블을 생성합니다. 이 모델은 좋아 보입니다!](/images/integrations/pytorch_example_table.png)

데이터셋과 모델을 로깅하고 시각화하는 방법에 대한 자세한 내용은 [W&B Tables 가이드](../tables/intro.md)를 확인하세요.

## PyTorch 코드 프로파일링하기

![W&B 대시보드 내부에서 PyTorch 코드 실행의 자세한 추적을 볼 수 있습니다.](/images/integrations/pytorch_example_dashboard.png)

W&B는 PyTorch 코드를 프로파일링하고 CPU 및 GPU 통신의 세부 사항을 검사하며 병목 현상을 식별하고 최적화를 위한 도구를 제공하기 위해 [PyTorch Kineto](https://github.com/pytorch/kineto)의 [Tensorboard 플러그인](https://github.com/pytorch/kineto/blob/master/tb\_plugin/README.md)과 직접 통합됩니다.

```python
profile_dir = "path/to/run/tbprofile/"
profiler = torch.profiler.profile(
    schedule=schedule,  # 스케줄에 대한 세부 정보는 프로파일러 문서를 참조하세요
    on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir),
    with_stack=True,
)

with profiler:
    ...  # 프로파일링하려는 코드를 여기서 실행하세요
    # 자세한 사용 방법은 프로파일러 문서를 참조하세요

# wandb Artifact 생성하기
profile_art = wandb.Artifact("trace", type="profile")
# Artifact에 pt.trace.json 파일 추가하기
profile_art.add_file(glob.glob(profile_dir + ".pt.trace.json"))
# artifact 로깅하기
profile_art.save()
```

[이 Colab](http://wandb.me/trace-colab)에서 작동하는 예제 코드를 보고 실행하세요.

:::caution
대화형 추적 보기 도구는 Chrome 추적 뷰어를 기반으로 하며 Chrome 브라우저에서 가장 잘 작동합니다.
:::