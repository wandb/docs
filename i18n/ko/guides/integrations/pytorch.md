---
displayed_sidebar: default
---

# PyTorch

[**여기에서 Colab 노트북으로 시도해 보세요 →**](http://wandb.me/intro)

PyTorch는 연구자들 사이에서 특히 인기 있는, 파이썬에서 딥 러닝을 위한 가장 인기 있는 프레임워크 중 하나입니다. W&B는 PyTorch에 대한 일급 지원을 제공하며, 그레이디언트 로깅부터 CPU 및 GPU에서 코드 프로파일링에 이르기까지 모두 지원합니다.

:::info
[colab 노트북](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Simple\_PyTorch\_Integration.ipynb)에서 우리의 통합을 시도해 보세요(아래에 비디오 튜토리얼 포함) 또는 [Hyperband](https://arxiv.org/abs/1603.06560)를 사용한 하이퍼파라미터 최적화를 포함한 스크립트를 위한 [예제 리포지토리](https://github.com/wandb/examples)를 확인하세요. [Fashion MNIST](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cnn-fashion) 및 생성된 [W&B 대시보드](https://wandb.ai/wandb/keras-fashion-mnist/runs/5z1d85qs)도 함께 볼 수 있습니다.
:::

## `wandb.watch`를 사용하여 그레이디언트 로깅하기

자동으로 그레이디언트를 로깅하려면, [`wandb.watch`](../../ref/python/watch.md)를 호출하고 PyTorch 모델을 전달하면 됩니다.

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
그레이디언트, 메트릭 및 그래프는 forward _and_ backward 패스가 호출된 후 `wandb.log`가 호출될 때까지 로깅되지 않습니다.
:::

## 이미지 및 미디어 로깅하기

PyTorch `Tensors`에 이미지 데이터를 [`wandb.Image`](../../ref/python/data-types/image.md)로 전달하면, [`torchvision`](https://pytorch.org/vision/stable/index.html)의 유틸리티가 자동으로 이미지로 변환됩니다:

```python
images_t = ...  # PyTorch Tensors로 이미지 생성 또는 로드
wandb.log({"examples": [wandb.Image(im) for im in images_t]})
```

PyTorch 및 기타 프레임워크에서 W&B로 리치 미디어를 로깅하는 방법에 대한 자세한 내용은 [미디어 로깅 가이드](../track/log/media.md)를 확인하세요.

또한 모델의 예측값이나 파생 메트릭과 같이 미디어와 함께 정보를 포함시키고 싶다면, `wandb.Table`을 사용하세요.

```python
my_table = wandb.Table()

my_table.add_column("image", images_t)
my_table.add_column("label", labels)
my_table.add_column("class_prediction", predictions_t)

# W&B에 테이블 로깅하기
wandb.log({"mnist_predictions": my_table})
```

![위 코드는 이와 같은 테이블을 생성합니다. 이 모델은 잘 보입니다!](/images/integrations/pytorch_example_table.png)

데이터세트와 모델을 로깅하고 시각화하는 방법에 대한 자세한 내용은 [W&B 테이블 가이드](../tables/intro.md)를 확인하세요.

## PyTorch 코드 프로파일링하기

![W&B 대시보드 내에서 PyTorch 코드 실행의 상세한 추적을 볼 수 있습니다.](/images/integrations/pytorch_example_dashboard.png)

W&B는 [PyTorch Kineto](https://github.com/pytorch/kineto)의 [Tensorboard 플러그인](https://github.com/pytorch/kineto/blob/master/tb\_plugin/README.md)과 직접 통합하여, PyTorch 코드 프로파일링 도구를 제공하고, CPU 및 GPU 통신의 세부 사항을 검사하며, 병목 현상과 최적화를 식별할 수 있습니다.

```python
profile_dir = "path/to/run/tbprofile/"
profiler = torch.profiler.profile(
    schedule=schedule,  # 프로파일러 문서에서 스케줄링에 대한 자세한 내용을 확인
    on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir),
    with_stack=True,
)

with profiler:
    ...  # 프로파일링하고 싶은 코드를 여기서 실행
    # 프로파일러 문서에서 자세한 사용법 정보를 확인

# W&B 아티팩트 생성
profile_art = wandb.Artifact("trace", type="profile")
# 아티팩트에 pt.trace.json 파일 추가
profile_art.add_file(glob.glob(profile_dir + ".pt.trace.json"))
# 아티팩트 로깅
profile_art.save()
```

[이 Colab](http://wandb.me/trace-colab)에서 작동하는 예제 코드를 보고 실행하세요.

:::caution
대화형 추적 보기 도구는 Chrome Trace Viewer를 기반으로 하며, Chrome 브라우저에서 가장 잘 작동합니다.
:::