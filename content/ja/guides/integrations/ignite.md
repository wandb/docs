---
title: PyTorch Ignite
description: W&B を PyTorch Ignite と統合する方法
menu:
  default:
    identifier: ignite
    parent: integrations
weight: 330
---

* この [W&B レポートの例 →](https://app.wandb.ai/example-team/pytorch-ignite-example/reports/PyTorch-Ignite-with-W%26B--Vmlldzo0NzkwMg) で、生成される可視化を確認できます
* この [ノートブックのホスティング例 →](https://colab.research.google.com/drive/15e-yGOvboTzXU4pe91Jg-Yr7sae3zBOJ#scrollTo=ztVifsYAmnRr) で、ご自身でコードの実行を試すことができます

Ignite は W&B ハンドラーをサポートしており、トレーニングと検証中にメトリクス、モデル／オプティマイザのパラメータ、勾配をログできます。また、モデルのチェックポイントを W&B クラウドにログすることも可能です。このクラスは wandb モジュールのラッパーにもなっています。つまり、このラッパーを使えば任意の wandb 関数を呼び出すことができます。モデルパラメータや勾配を保存する例については、下記をご覧ください。

## 基本セットアップ

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
    # データの前処理を定義
    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    # トレーニングデータローダーを作成
    train_loader = DataLoader(MNIST(download=True, root=".", transform=data_transform, train=True),
                              batch_size=train_batch_size, shuffle=True)

    # 検証データローダーを作成
    val_loader = DataLoader(MNIST(download=False, root=".", transform=data_transform, train=False),
                            batch_size=val_batch_size, shuffle=False)
    return train_loader, val_loader
```

ignite で `WandBLogger` を使うのは柔軟なステップです。まず、`WandBLogger` オブジェクトを作成します。次に、それを trainer または evaluator にアタッチすることで、自動的にメトリクスをログできます。この例では以下を示しています:

* トレーニングロスを trainer オブジェクトにアタッチしてログ
* 検証ロスを evaluator にアタッチしてログ
* 学習率などのオプションパラメータもログ
* モデルをウォッチ

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
    # WandBLogger オブジェクトの作成
    wandb_logger = WandBLogger(
    project="pytorch-ignite-integration",
    name="cnn-mnist",
    config={"max_epochs": epochs,"batch_size":train_batch_size},
    tags=["pytorch-ignite", "mninst"]
    )

    # Trainer へトレーニングロスを紐付けてログ
    wandb_logger.attach_output_handler(
    trainer,
    event_name=Events.ITERATION_COMPLETED,
    tag="training",
    output_transform=lambda loss: {"loss": loss}
    )

    # Evaluator へ検証ロスを紐付けてログ
    wandb_logger.attach_output_handler(
    evaluator,
    event_name=Events.EPOCH_COMPLETED,
    tag="training",
    metric_names=["nll", "accuracy"],
    global_step_transform=lambda *_: trainer.state.iteration,
    )

    # オプティマイザーのパラメータ（例: 学習率）をログ
    wandb_logger.attach_opt_params_handler(
    trainer,
    event_name=Events.ITERATION_STARTED,
    optimizer=optimizer,
    param_name='lr'  # 任意
    )

    # モデルをウォッチ
    wandb_logger.watch(model)
```

必要に応じて ignite の `EVENTS` を利用し、メトリクスを直接ターミナルに出力することもできます

```python
    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def log_training_loss(engine):
        # トレーニングロスの進捗を表示
        pbar.desc = desc.format(engine.state.output)
        pbar.update(log_interval)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        # エポックごとのトレーニング結果を表示
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
        # エポックごとの検証結果を表示
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
                        help='トレーニング用の入力バッチサイズ (デフォルト: 64)')
    parser.add_argument('--val_batch_size', type=int, default=1000,
                        help='検証用の入力バッチサイズ (デフォルト: 1000)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='トレーニングするエポック数 (デフォルト: 10)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='学習率 (デフォルト: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD モーメンタム (デフォルト: 0.5)')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='トレーニングステータスをログするバッチ間隔')

    args = parser.parse_args()
    run(args.batch_size, args.val_batch_size, args.epochs, args.lr, args.momentum, args.log_interval)
```

このコードにより、次のような可視化が生成されます::

{{< img src="/images/integrations/pytorch-ignite-1.png" alt="PyTorch Ignite トレーニングダッシュボード" >}}

{{< img src="/images/integrations/pytorch-ignite-2.png" alt="PyTorch Ignite パフォーマンス" >}}

{{< img src="/images/integrations/pytorch-ignite-3.png" alt="PyTorch Ignite ハイパーパラメータチューニング結果" >}}

{{< img src="/images/integrations/pytorch-ignite-4.png" alt="PyTorch Ignite モデル比較ダッシュボード" >}}

詳細については [Ignite ドキュメント](https://pytorch.org/ignite/contrib/handlers.html#module-ignite.contrib.handlers.wandb_logger) を参照してください。