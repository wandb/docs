---
slug: /guides/integrations/ignite
description: PyTorch Ignite と W&B の統合方法
---

# PyTorch Ignite

* 結果の可視化はこれらの[例のW&Bレポート →](https://app.wandb.ai/example-team/pytorch-ignite-example/reports/PyTorch-Ignite-with-W%26B--Vmlldzo0NzkwMg)で見ることができます。
* この[ホストされたノートブックの例 →](https://colab.research.google.com/drive/15e-yGOvboTzXU4pe91Jg-Yr7sae3zBOJ#scrollTo=ztVifsYAmnRr)で自分でコードを実行してみてください。

Igniteは、トレーニングと検証中にメトリクス、モデル/オプティマイザーパラメータ、勾配をログに記録するためのWeights & Biasesハンドラーをサポートしています。また、Weights & Biases クラウドにモデルのチェックポイントもログに記録することができます。このクラスは、wandbモジュールのラッパーでもあります。つまり、このラッパーを使って任意のwandb関数を呼び出すことができます。モデルパラメータと勾配を保存する方法の例を参照してください。

## 基本的な PyTorch セットアップ

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
```
以下は翻訳するMarkdownテキストのチャンクです。日本語に翻訳してください。それ以外のことは何も言わずに、翻訳されたテキストのみを返してください。テキスト：

```python
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

ignite で WandBLogger を使うことは、2 ステップのモジュール化されたプロセスです。まず、WandBLogger オブジェクトを作成する必要があります。その後、任意のトレーナーや評価者に自動的にメトリクスを記録するためにそれをアタッチすることができます。以下のタスクを順に実行します。
1) WandBLogger オブジェクトを作成する
2) オブジェクトを出力ハンドラにアタッチして以下を行う:

* トレーニングの損失を記録 - トレーナーオブジェクトにアタッチ
* 検証損失の記録 - 評価者にアタッチ
* 任意のパラメータを記録する - 例えば、学習率
* モデルを監視する

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
    #WandBloggerオブジェクトの作成
    wandb_logger = WandBLogger(
    project="pytorch-ignite-integration",
    name="cnn-mnist",
    config={"max_epochs": epochs,"batch_size":train_batch_size},
    tags=["pytorch-ignite", "mninst"]
    )
以下は、Markdownのテキストチャンクを翻訳してください。日本語に翻訳し、翻訳したテキストだけを返してください。他のことは言わないでください。テキスト：

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
    param_name='lr'  # optional
    )

    wandb_logger.watch(model)
```

オプションで、igniteの `EVENTS`を使ってメトリクスを直接ターミナルにログにもできます。

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
            "トレーニング結果 - エポック: {}  平均精度: {:.2f} 平均損失: {:.2f}"
            .format(engine.state.epoch, avg_accuracy, avg_nll)
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        tqdm.write(
            "検証結果 - エポック: {}  平均精度: {:.2f} 平均損失: {:.2f}"
            .format(engine.state.epoch, avg_accuracy, avg_nll))

        pbar.n = pbar.last_print_n = 0

    trainer.run(train_loader, max_epochs=epochs)
    pbar.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64,
                        help='トレーニング用の入力バッチサイズ（デフォルト：64）')
    parser.add_argument('--val_batch_size', type=int, default=1000,
                        help='検証用の入力バッチサイズ（デフォルト：1000）')
    parser.add_argument('--epochs', type=int, default=10,
                        help='トレーニングするエポック数（デフォルト：10）')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='学習率（デフォルト：0.01）')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGDの運動量（デフォルト：0.5）')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='トレーニングステータスをログに記録する前に待機するバッチ数')
以下のMarkdownテキストを日本語に翻訳してください。翻訳したテキストのみを返し、他のことは何も言わないでください。テキスト：

args = parser.parse_args()

    run(args.batch_size, args.val_batch_size, args.epochs, args.lr, args.momentum, args.log_interval)

```

上記のコードを実行すると、以下のデータ可視化が得られます。

![](https://i.imgur.com/CoBDShx.png)

![](https://i.imgur.com/Fr6Dqd0.png)

![](https://i.imgur.com/Fr6Dqd0.png)

![](https://i.imgur.com/rHNPyw3.png)

より詳細なドキュメントについては、[Ignite Docs](https://pytorch.org/ignite/contrib/handlers.html#module-ignite.contrib.handlers.wandb_logger)を参照してください。