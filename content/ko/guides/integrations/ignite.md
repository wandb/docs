---
title: PyTorch Ignite
description: W&B를 PyTorch Ignite와 통합하는 방법
menu:
  default:
    identifier: ko-guides-integrations-ignite
    parent: integrations
weight: 330
---

* 이 [예시 W&B report →](https://app.wandb.ai/example-team/pytorch-ignite-example/reports/PyTorch-Ignite-with-W%26B--Vmlldzo0NzkwMg)에서 결과 시각화를 확인해보세요.
* 이 [예시 호스팅된 노트북 →](https://colab.research.google.com/drive/15e-yGOvboTzXU4pe91Jg-Yr7sae3zBOJ#scrollTo=ztVifsYAmnRr)에서 직접 코드를 실행할 수 있습니다.

Ignite는 W&B handler를 지원하여 트레이닝과 검증 과정에서 메트릭, 모델/옵티마이저 파라미터, 그레이디언트 등을 로그로 남길 수 있습니다. 또한, 모델 체크포인트를 W&B 클라우드에 저장하는 데에도 사용할 수 있습니다. 이 클래스는 wandb 모듈의 래퍼이기 때문에, 해당 래퍼를 통해 어떤 wandb 함수든 호출할 수 있습니다. 모델 파라미터와 그레이디언트를 저장하는 예제는 아래에서 확인하실 수 있습니다.

## 기본 설정

```python
# 필요한 라이브러리 임포트
from argparse import ArgumentParser
import wandb
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import MNIST

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss

from tqdm import tqdm


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


def get_data_loaders(train_batch_size, val_batch_size):
    # 데이터 전처리 및 DataLoader 생성
    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    train_loader = DataLoader(MNIST(download=True, root=".", transform=data_transform, train=True),
                              batch_size=train_batch_size, shuffle=True)

    val_loader = DataLoader(MNIST(download=False, root=".", transform=data_transform, train=False),
                            batch_size=val_batch_size, shuffle=False)
    return train_loader, val_loader
```

Ignite에서 `WandBLogger`를 사용하는 방법은 모듈화되어 있습니다. 먼저 `WandBLogger` 오브젝트를 생성한 뒤, 트레이너나 평가자에 attach해서 메트릭을 자동으로 로그할 수 있습니다. 이 예제에서는 아래와 같이 동작합니다.

* 트레이닝 손실(loss)을 trainer 오브젝트에 attach해서 로그합니다.
* 평가 손실(검증 loss)을 evaluator에 attach해서 로그합니다.
* 학습률(learning rate)과 같은 추가 파라미터도 로그합니다.
* 모델을 watch합니다.

```python
from ignite.contrib.handlers.wandb_logger import *
def run(train_batch_size, val_batch_size, epochs, lr, momentum, log_interval):
    train_loader, val_loader = get_data_loaders(train_batch_size, val_batch_size)
    model = Net()
    device = 'cpu'

    if torch.cuda.is_available():
        device = 'cuda'

    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    trainer = create_supervised_trainer(model, optimizer, F.nll_loss, device=device)
    evaluator = create_supervised_evaluator(model,
                                            metrics={'accuracy': Accuracy(),
                                                     'nll': Loss(F.nll_loss)},
                                            device=device)

    desc = "ITERATION - loss: {:.2f}"
    pbar = tqdm(
        initial=0, leave=False, total=len(train_loader),
        desc=desc.format(0)
    )
    # WandBLogger 오브젝트 생성
    wandb_logger = WandBLogger(
    project="pytorch-ignite-integration",
    name="cnn-mnist",
    config={"max_epochs": epochs,"batch_size":train_batch_size},
    tags=["pytorch-ignite", "mninst"]
    )

    wandb_logger.attach_output_handler(
    trainer,
    event_name=Events.ITERATION_COMPLETED,
    tag="training",
    output_transform=lambda loss: {"loss": loss}
    )

    wandb_logger.attach_output_handler(
    evaluator,
    event_name=Events.EPOCH_COMPLETED,
    tag="training",
    metric_names=["nll", "accuracy"],
    global_step_transform=lambda *_: trainer.state.iteration,
    )

    wandb_logger.attach_opt_params_handler(
    trainer,
    event_name=Events.ITERATION_STARTED,
    optimizer=optimizer,
    param_name='lr'  # 옵션
    )

    wandb_logger.watch(model)
```

원한다면 ignite의 `EVENTS`를 직접 활용해 메트릭을 터미널에 바로 출력할 수도 있습니다.

```python
    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def log_training_loss(engine):
        pbar.desc = desc.format(engine.state.output)
        pbar.update(log_interval)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        pbar.refresh()
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        tqdm.write(
            "Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
            .format(engine.state.epoch, avg_accuracy, avg_nll)
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        tqdm.write(
            "Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
            .format(engine.state.epoch, avg_accuracy, avg_nll))

        pbar.n = pbar.last_print_n = 0

    trainer.run(train_loader, max_epochs=epochs)
    pbar.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64,
                        help='트레이닝에 사용할 입력 배치 크기 (기본값: 64)')
    parser.add_argument('--val_batch_size', type=int, default=1000,
                        help='검증에 사용할 입력 배치 크기 (기본값: 1000)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='트레이닝할 에포크 수 (기본값: 10)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='학습률 (기본값: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD 모멘텀 (기본값: 0.5)')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='트레이닝 상태를 로그하기 전까지 기다릴 배치 수')

    args = parser.parse_args()
    run(args.batch_size, args.val_batch_size, args.epochs, args.lr, args.momentum, args.log_interval)
```

이 코드는 다음과 같은 시각화 결과를 생성합니다:

{{< img src="/images/integrations/pytorch-ignite-1.png" alt="PyTorch Ignite 트레이닝 대시보드" >}}

{{< img src="/images/integrations/pytorch-ignite-2.png" alt="PyTorch Ignite 성능" >}}

{{< img src="/images/integrations/pytorch-ignite-3.png" alt="PyTorch Ignite 하이퍼파라미터 튜닝 결과" >}}

{{< img src="/images/integrations/pytorch-ignite-4.png" alt="PyTorch Ignite 모델 비교 대시보드" >}}

더 자세한 내용은 [Ignite Docs](https://pytorch.org/ignite/contrib/handlers.html#module-ignite.contrib.handlers.wandb_logger)를 참고하세요.