---
description: How to integrate W&B with PyTorch Ignite.
slug: /guides/integrations/ignite
displayed_sidebar: default
---

# PyTorch Ignite

* 이 [예제 W&B 리포트 →](https://app.wandb.ai/example-team/pytorch-ignite-example/reports/PyTorch-Ignite-with-W%26B--Vmlldzo0NzkwMg)에서 결과 시각화를 확인하세요.
* 이 [예제 호스팅 노트북 →](https://colab.research.google.com/drive/15e-yGOvboTzXU4pe91Jg-Yr7sae3zBOJ#scrollTo=ztVifsYAmnRr)에서 코드를 직접 실행해 보세요.

Ignite는 메트릭, 모델/옵티마이저 파라미터, 트레이닝 및 검증 도중 그레이디언트를 로그하는 Weights & Biases 핸들러를 지원합니다. 또한 모델 체크포인트를 Weights & Biases 클라우드에 로그하는 데 사용할 수 있습니다. 이 클래스는 또한 wandb 모듈의 래퍼이기도 합니다. 이는 이 래퍼를 사용하여 wandb 함수를 호출할 수 있음을 의미합니다. 모델 파라미터와 그레이디언트를 저장하는 방법에 대한 예시를 참고하세요.

## 기본 PyTorch 설정

```python
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
    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    train_loader = DataLoader(MNIST(download=True, root=".", transform=data_transform, train=True),
                              batch_size=train_batch_size, shuffle=True)

    val_loader = DataLoader(MNIST(download=False, root=".", transform=data_transform, train=False),
                            batch_size=val_batch_size, shuffle=False)
    return train_loader, val_loader
```

Ignite에서 WandBLogger를 사용하는 것은 2단계 모듈 프로세스입니다: 먼저, WandBLogger 객체를 생성해야 합니다. 그런 다음 트레이너 또는 평가자에게 자동으로 메트릭을 로그하기 위해 첨부할 수 있습니다. 다음 작업을 순차적으로 수행하겠습니다: 1) WandBLogger 객체 생성 2) 다음 작업에 객체를 첨부합니다:

* 트레이닝 손실 로그 - 트레이너 객체에 첨부
* 검증 손실 로그 - 평가자에 첨부
* 선택적 파라미터 로그 - 예를 들어, 학습률
* 모델 관찰

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
    #WandBlogger 객체 생성
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
    param_name='lr'  # 선택적
    )

    wandb_logger.watch(model)
```

선택적으로, ignite `EVENTS`를 사용하여 메트릭을 직접 터미널에 로그할 수도 있습니다.

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
            "트레이닝 결과 - 에포크: {}  평균 정확도: {:.2f} 평균 손실: {:.2f}"
            .format(engine.state.epoch, avg_accuracy, avg_nll)
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        tqdm.write(
            "검증 결과 - 에포크: {}  평균 정확도: {:.2f} 평균 손실: {:.2f}"
            .format(engine.state.epoch, avg_accuracy, avg_nll))

        pbar.n = pbar.last_print_n = 0

    trainer.run(train_loader, max_epochs=epochs)
    pbar.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64,
                        help='트레이닝을 위한 입력 배치 크기 (기본값: 64)')
    parser.add_argument('--val_batch_size', type=int, default=1000,
                        help='검증을 위한 입력 배치 크기 (기본값: 1000)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='트레이닝할 에포크 수 (기본값: 10)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='학습률 (기본값: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD 모멘텀 (기본값: 0.5)')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='트레이닝 상태를 로깅하기 전에 기다릴 배치 수')

    args = parser.parse_args()
    run(args.batch_size, args.val_batch_size, args.epochs, args.lr, args.momentum, args.log_interval)
```

위 코드를 실행하면 다음과 같은 시각화를 얻을 수 있습니다:

![](https://i.imgur.com/CoBDShx.png)

![](https://i.imgur.com/Fr6Dqd0.png)

![](https://i.imgur.com/Fr6Dqd0.png)

![](https://i.imgur.com/rHNPyw3.png)

더 자세한 문서는 [Ignite Docs](https://pytorch.org/ignite/contrib/handlers.html#module-ignite.contrib.handlers.wandb_logger)를 참조하세요