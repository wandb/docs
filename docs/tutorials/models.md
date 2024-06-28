
# Register models

[**Try in a Colab Notebook here →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-model-registry/Model_Registry_E2E.ipynb)

モデルレジストリは、組織全体で現在作業中のすべてのモデルタスクと関連するArtifactsを集めて整理するための中央の場所です。
- モデルチェックポイント管理
- リッチなモデルカードでモデルを文書化
- 使用中/デプロイメント中のすべてのモデルの履歴を維持
- モデルのクリーンな引継ぎとステージ管理の促進
- 各種モデルタスクのタグ付けと整理
- モデルの進捗時に自動通知を設定

このチュートリアルでは、簡単な画像分類タスクのモデル開発ライフサイクルを追跡する方法をご紹介します。

### 🛠️ `wandb` のインストール

```bash
!pip install -q wandb onnx pytorch-lightning
```

## W&B にログイン
- `wandb login` または `wandb.login()` を使用して明示的にログインできます（以下参照）
- あるいは環境変数を設定することも可能です。W&Bロギングの振る舞いを変更するために設定できる環境変数は多岐にわたり、最も重要なものは以下の通りです:
    - `WANDB_API_KEY` - プロファイルの「Settings」セクションで見つかります
    - `WANDB_BASE_URL` - W&BサーバーのURLです
- APIトークンはW&Bアプリの「Profile」 -> 「Settings」で見つかります

![api_token](https://drive.google.com/uc?export=view&id=1Xn7hnn0rfPu_EW0A_-32oCXqDmpA0-kx)

```notebook
!wandb login
```

:::note
[W&B Server](..//guides/hosting/intro.md)デプロイメント（**Dedicated Cloud**または**Self-managed**のどちらか）に接続する際には、--reloginおよび--hostオプションを使用します。例えば:

```notebook
!wandb login --relogin --host=http://your-shared-local-host.com
```

必要に応じて、デプロイメント管理者にホスト名を確認してください。
:::

## Log Data and Model Checkpoints as Artifacts
W&B Artifactsを使用すると、データセット、モデルチェックポイント、評価結果などの任意のシリアライズデータを追跡しバージョン管理が可能です。アーティファクトを作成する際には名前とタイプを指定します。このアーティファクトは実験的なSoR（system of record）と永久にリンクされます。基礎データが変更され、そのデータ資産を再度ログに記録すると、W&Bが自動的に内容をチェックサムし、新しいバージョンを作成します。W&B Artifactsは、共有の非構造化ファイルシステムの上の軽量な抽象レイヤーとして考えることができます。

### アーティファクトの構造

`Artifact`クラスは、W&Bアーティファクトレジストリ内のエントリに対応します。アーティファクトには以下が含まれます:
* 名前
* タイプ
* メタデータ
* 説明
* ファイル、およびディレクトリ、または参照

使用例:
```python
run = wandb.init(project="my-project")
artifact = wandb.Artifact(name="my_artifact", type="data")
artifact.add_file("/path/to/my/file.txt")
run.log_artifact(artifact)
run.finish()
```

このチュートリアルでは、まずトレーニングデータセットをダウンロードし、それをトレーニングジョブで使用するアーティファクトとしてログに記録します。

```python
# @title Enter your W&B project and entity

# FORM VARIABLES
PROJECT_NAME = "model-registry-tutorial"  # @param {type:"string"}
ENTITY = None  # @param {type:"string"}

# SIZEを"TINY"、"SMALL"、"MEDIUM"、または"LARGE"に設定して、以下のデータセットのいずれかを選択
# TINYデータセット: 100枚の画像、30MB
# SMALLデータセット: 1000枚の画像、312MB
# MEDIUMデータセット: 5000枚の画像、1.5GB
# LARGEデータセット: 12,000枚の画像、3.6GB

SIZE = "TINY"

if SIZE == "TINY":
    src_url = "https://storage.googleapis.com/wandb_datasets/nature_100.zip"
    src_zip = "nature_100.zip"
    DATA_SRC = "nature_100"
    IMAGES_PER_LABEL = 10
    BALANCED_SPLITS = {"train": 8, "val": 1, "test": 1}
elif SIZE == "SMALL":
    src_url = "https://storage.googleapis.com/wandb_datasets/nature_1K.zip"
    src_zip = "nature_1K.zip"
    DATA_SRC = "nature_1K"
    IMAGES_PER_LABEL = 100
    BALANCED_SPLITS = {"train": 80, "val": 10, "test": 10}
elif SIZE == "MEDIUM":
    src_url = "https://storage.googleapis.com/wandb_datasets/nature_12K.zip"
    src_zip = "nature_12K.zip"
    DATA_SRC = "inaturalist_12K/train"  #（技術的には10,000枚の画像のみのサブセット）
    IMAGES_PER_LABEL = 500
    BALANCED_SPLITS = {"train": 400, "val": 50, "test": 50}
elif SIZE == "LARGE":
    src_url = "https://storage.googleapis.com/wandb_datasets/nature_12K.zip"
    src_zip = "nature_12K.zip"
    DATA_SRC = "inaturalist_12K/train"  #（技術的には10,000枚の画像のみのサブセット）
    IMAGES_PER_LABEL = 1000
    BALANCED_SPLITS = {"train": 800, "val": 100, "test": 100}
```

```notebook
%%capture
!curl -SL $src_url > $src_zip
!unzip $src_zip
```

```python
import wandb
import pandas as pd
import os

with wandb.init(project=PROJECT_NAME, entity=ENTITY, job_type="log_datasets") as run:
    img_paths = []
    for root, dirs, files in os.walk("nature_100", topdown=False):
        for name in files:
            img_path = os.path.join(root, name)
            label = img_path.split("/")[1]
            img_paths.append([img_path, label])

    index_df = pd.DataFrame(columns=["image_path", "label"], data=img_paths)
    index_df.to_csv("index.csv", index=False)

    train_art = wandb.Artifact(
        name="Nature_100",
        type="raw_images",
        description="nature image dataset with 10 classes, 10 images per class",
    )
    train_art.add_dir("nature_100")

    # 各画像のラベルを示すcsvも追加
    train_art.add_file("index.csv")
    wandb.log_artifact(train_art)
```

### アーティファクト名とエイリアスを使用してデータ資産を簡単に引き継ぎおよび抽象化
- データセットやモデルの`name:alias`組み合わせを参照するだけで、ワークフローコンポーネントをより良く標準化できます 
- 例として、PyTorch `Dataset`や`DataModule`を構築し、引数としてW&B Artifact名とエイリアスを使用して適切にロードできます

このデータセットに関連するすべてのメタデータ、これを使用するW&B runs、および上下流の全アーティファクトのリネージを今すぐ確認できます！

![api_token](https://drive.google.com/uc?export=view&id=1fEEddXMkabgcgusja0g8zMz8whlP2Y5P)

```python
from torchvision import transforms
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from skimage import io, transform
from torchvision import transforms, utils, models
import math

class NatureDataset(Dataset):
    def __init__(
        self,
        wandb_run,
        artifact_name_alias="Nature_100:latest",
        local_target_dir="Nature_100:latest",
        transform=None,
    ):
        self.local_target_dir = local_target_dir
        self.transform = transform

        # アーティファクトをローカルに取得してメモリにロード
        art = wandb_run.use_artifact(artifact_name_alias)
        path_at = art.download(root=self.local_target_dir)

        self.ref_df = pd.read_csv(os.path.join(self.local_target_dir, "index.csv"))
        self.class_names = self.ref_df.iloc[:, 1].unique().tolist()
        self.idx_to_class = {k: v for k, v in enumerate(self.class_names)}
        self.class_to_idx = {v: k for k, v in enumerate(self.class_names)}

    def __len__(self):
        return len(self.ref_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.ref_df.iloc[idx, 0]

        image = io.imread(img_path)
        label = self.ref_df.iloc[idx, 1]
        label = torch.tensor(self.class_to_idx[label], dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label

class NatureDatasetModule(pl.LightningDataModule):
    def __init__(
        self,
        wandb_run,
        artifact_name_alias: str = "Nature_100:latest",
        local_target_dir: str = "Nature_100:latest",
        batch_size: int = 16,
        input_size: int = 224,
        seed: int = 42,
    ):
        super().__init__()
        self.wandb_run = wandb_run
        self.artifact_name_alias = artifact_name_alias
        self.local_target_dir = local_target_dir
        self.batch_size = batch_size
        self.input_size = input_size
        self.seed = seed

    def setup(self, stage=None):
        self.nature_dataset = NatureDataset(
            wandb_run=self.wandb_run,
            artifact_name_alias=self.artifact_name_alias,
            local_target_dir=self.local_target_dir,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.CenterCrop(self.input_size),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            ),
        )

        nature_length = len(self.nature_dataset)
        train_size = math.floor(0.8 * nature_length)
        val_size = math.floor(0.2 * nature_length)
        self.nature_train, self.nature_val = random_split(
            self.nature_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.seed),
        )
        return self

    def train_dataloader(self):
        return DataLoader(self.nature_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.nature_val, batch_size=self.batch_size)

    def predict_dataloader(self):
        pass

    def teardown(self, stage: str):
        pass
```

## Model Training

### モデルクラスと検証関数の作成

```python
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import onnx

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # このifステートメントで設定される変数を初期化します。これらの変数はそれぞれのモデルに特有です。
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """Resnet18"""
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = torch.nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """Alexnet"""
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = torch.nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """VGG11_bn"""
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = torch.nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """Squeezenet"""
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = torch.nn.Conv2d(
            512, num_classes, kernel_size=(1, 1), stride=(1, 1)
        )
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """Densenet"""
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = torch.nn.Linear(num_ftrs, num_classes)
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

class NaturePyTorchModule(torch.nn.Module):
    def __init__(self, model_name, num_classes=10, feature_extract=True, lr=0.01):
        """モデルパラメータを定義するためのメソッド"""
        super().__init__()

        self.model_name = model_name
        self.num_classes = num_classes
        self.feature_extract = feature_extract
        self.lr = lr
        self.model, self.input_size = initialize_model(
            model_name=self.model_name,
            num_classes=self.num_classes,
            feature_extract=True,
        )

    def forward(self, x):
        """推論用の入力 -> 出力メソッド"""
        x = self.model(x)

        return x

def evaluate_model(model, eval_data, idx_to_class, class_names, epoch_ndx):
    device = torch.device("cpu")
    model.eval()
    test_loss = 0
    correct = 0
    preds = []
    actual = []

    val_table = wandb.Table(columns=["pred", "actual", "image"])

    with torch.no_grad():
        for data, target in eval_data:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # バッチ損失の合計
            pred = output.argmax(
                dim=1, keepdim=True
            )  # 最大対数確率のインデックスを取得
            preds += list(pred.flatten().tolist())
            actual += target.numpy().tolist()
            correct += pred.eq(target.view_as(pred)).sum().item()

            for idx, img in enumerate(data):
                img = img.numpy().transpose(1, 2, 0)
                pred_class = idx_to_class[pred.numpy()[idx][0]]
                target_class = idx_to_class[target.numpy()[idx]]
                val_table.add_data(pred_class, target_class, wandb.Image(img))

    test_loss /= len(eval_data.dataset)
    accuracy = 100.0 * correct / len(eval_data.dataset)
    conf_mat = wandb.plot.confusion_matrix(
        y_true=actual, preds=preds, class_names=class_names
    )
    return test_loss, accuracy, preds, val_table, conf_mat
```

### トレーニングのループの追跡
トレーニング中には、定期的にモデルのチェックポイントを作成するのがベストプラクティスです。こうすることで、トレーニングが中断されたり、インスタンスがクラッシュした場合でも、途中から再開できます。Artifactをログすることで、すべてのチェックポイントをW&Bで追跡し、シリアライゼーションの形式やクラスラベルなどのメタデータを付加できます。これにより、誰かがチェックポイントを利用する際に、どのように使用すればよいかが分かります。Artifactとして任意の形式のモデルをログする際には、Artifactの`type`を`model`に設定してください。

```python
run = wandb.init(
    project=PROJECT_NAME,
    entity=ENTITY,
    job_type="training",
    config={
        "model_type": "squeezenet",
        "lr": 1.0,
        "gamma": 0.75,
        "batch_size": 16,
        "epochs": 5,
    },
)

model = NaturePyTorchModule(wandb.config["model_type"])
wandb.watch(model)

wandb.config["input_size"] = 224

nature_module = NatureDatasetModule(
    wandb_run=run,
    artifact_name_alias="Nature_100:latest",
    local_target_dir="Nature_100:latest",
    batch_size=wandb.config["batch_size"],
    input_size=wandb.config["input_size"],
)
nature_module.setup()

# モデルのトレーニング
learning_rate = wandb.config["lr"]
gamma = wandb.config["gamma"]
epochs = wandb.config["epochs"]

device = torch.device("cpu")
optimizer = optim.Adadelta(model.parameters(), lr=wandb.config["lr"])
scheduler = StepLR(optimizer, step_size=1, gamma=wandb.config["gamma"])

best_loss = float("inf")
best_model = None

for epoch_ndx in range(epochs):
    model.train()
    for batch_ndx, batch in enumerate(nature_module.train_dataloader()):
        data, target = batch[0].to("cpu"), batch[1].to("cpu")
        optimizer.zero_grad()
        preds = model(data)
        loss = F.nll_loss(preds, target)
        loss.backward()
        optimizer.step()
        scheduler.step()

        ### あなたのメトリクスをログしましょう ###
        wandb.log(
            {
                "train/epoch_ndx": epoch_ndx,
                "train/batch_ndx": batch_ndx,
                "train/train_loss": loss,
                "train/learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

    ### 各エポック終了時の評価 ###
    model.eval()
    test_loss, accuracy, preds, val_table, conf_mat = evaluate_model(
        model,
        nature_module.val_dataloader(),
        nature_module.nature_dataset.idx_to_class,
        nature_module.nature_dataset.class_names,
        epoch_ndx,
    )

    is_best = test_loss < best_loss

    wandb.log(
        {
            "eval/test_loss": test_loss,
            "eval/accuracy": accuracy,
            "eval/conf_mat": conf_mat,
            "eval/val_table": val_table,
        }
    )

    ### モデルの重みをチェックポイントとして保存 ###
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    torch.onnx.export(
        model,  # 実行されるモデル
        x,  # モデルの入力（または複数入力の場合はタプル）
        "model.onnx",  # モデルを保存する場所（ファイルまたはファイルライクオブジェクト）
        export_params=True,  # トレーニング済みのパラメータ重量をモデルファイルに保存
        opset_version=10,  # モデルをエクスポートするONNXバージョン
        do_constant_folding=True,  # 最適化のための定数折りたたみを実行するかどうか
        input_names=["input"],  # モデルの入力名
        output_names=["output"],  # モデルの出力名
        dynamic_axes={
            "input": {0: "batch_size"},  # 可変長の軸
            "output": {0: "batch_size"},
        },
    )

    art = wandb.Artifact(
        f"nature-{wandb.run.id}",
        type="model",
        metadata={
            "format": "onnx",
            "num_classes": len(nature_module.nature_dataset.class_names),
            "model_type": wandb.config["model_type"],
            "model_input_size": wandb.config["input_size"],
            "index_to_class": nature_module.nature_dataset.idx_to_class,
        },
    )

    art.add_file("model.onnx")

    ### 長期にわたって最良のチェックポイントを追跡するエイリアスを追加 ###
    wandb.log_artifact(art, aliases=["best", "latest"] if is_best else None)
    if is_best:
        best_model = art
```

### 1つのプロジェクトの下であなたのすべてのモデルチェックポイントを管理します。

![api_token](https://drive.google.com/uc?export=view&id=1z7nXRgqHTPYjfR1SoP-CkezyxklbAZlM)

### 注記: W&Bオフラインとの同期
トレーニング中に何らかの理由でネットワーク通信が切断された場合でも、`wandb sync`を使用して進捗を常に同期できます。

W&B SDKはすべてのログデータをローカルディレクトリ`wandb`にキャッシュし、`wandb sync`を呼び出すと、ローカルの状態がウェブアプリと同期されます。

## モデルレジストリ
実験中に複数のrunで多数のチェックポイントをログした後は、次のワークフローステージ（例：テスト、デプロイメント）に最良のチェックポイントを引き継ぐ時です。

モデルレジストリは、個々のW&Bプロジェクトの上位に位置する中央ページです。ここには**Registered Models**が保存され、個々のW&Bプロジェクトに存在する価値のあるチェックポイントへの「リンク」を格納します。

モデルレジストリは、すべてのモデルタスクの最良のチェックポイントを保存するための集中管理の場を提供します。あなたがログする任意の`model`アーティファクトは、Registered Modelに「リンク」できます。

### **Registered Models**を作成し、UIからリンクする

#### 1. チームページにアクセスし、`Model Registry`を選択して、チームのモデルレジストリにアクセスします。

![model registry](https://drive.google.com/uc?export=view&id=1ZtJwBsFWPTm4Sg5w8vHhRpvDSeQPwsKw)

#### 2. 新しいRegistered Modelを作成します。

![model registry](https://drive.google.com/uc?export=view&id=1RuayTZHNE0LJCxt1t0l6-2zjwiV4aDXe)

#### 3. モデルのすべてのチェックポイントが格納されているプロジェクトのアーティファクトタブに移動します。

![model registry](https://drive.google.com/uc?export=view&id=1LfTLrRNpBBPaUb_RmBIE7fWFMG0h3e0E)

#### 4. モデルアーティファクトバージョンの「Link to Registry」をクリックします。

### **API**を通じてRegistered Modelsを作成しリンクする
`wandb.run.link_artifact`を使用してアーティファクトオブジェクトおよび**Registered Model**の名前、さらに付加したいエイリアスを渡して、[API経由でモデルをリンク](https://docs.wandb.ai/guides/models)することができます。W&Bでは**Registered Models**はエンティティ（チーム）にスコープされているため、チームのメンバーのみがそこにある**Registered Models**を表示およびアクセスできます。APIでRegistered Modelの名前を指定する場合は、`<entity>/model-registry/<registered-model-name>`とします。Registered Modelが存在しない場合、自動的に作成されます。

```python
if ENTITY:
    wandb.run.link_artifact(
        best_model,
        f"{ENTITY}/model-registry/Model Registry Tutorial",
        aliases=["staging"],
    )
else:
    print("Must indicate entity where Registered Model will exist")
wandb.finish()
```

### 「リンク」とは？
レジストリにリンクすると、そのRegistered Modelの新しいバージョンが作成されます。これはそのプロジェクト内に存在するアーティファクトバージョンへのポインタにすぎません。W&Bがプロジェクト内のアーティファクトのバージョン管理とRegistered Modelのバージョン管理を分ける理由はここにあります。モデルアーティファクトのバージョンをリンクするプロセスは、そのアーティファクトバージョンをRegistered Modelタスクの下で「ブックマーク」することと同等です。

通常、研究開発や実験中に研究者は100以上、さらには1000以上のモデルチェックポイントアーティファクトを生成しますが、そのうち実際に「日の目を見る」ものは一つか二つだけです。これらのチェックポイントを別のバージョン管理されたレジストリにリンクするプロセスは、モデル開発側とモデルデプロイメント/消費側のワークフローを明確に区別するのに役立ちます。モデルの世界共通のバージョン/エイリアスは、研究開発中に生成されるすべての実験バージョンから汚染されないようにすべきであり、したがって、Registered Modelのバージョン管理は新しい「ブックマークされた」モデルに応じて増加します。

## すべてのモデルのための集中ハブを作成
- Registered Modelにモデルカード、タグ、Slack通知を追加
- モデルが異なるフェーズを移行する際にエイリアスを変更
- モデルドキュメントと回帰レポートのためにモデルレジストリをレポートに埋め込みます。例えばこのレポートを参照してください：[example](https://api.wandb.ai/links/wandb-smle/r82bj9at)
![model registry](https://drive.google.com/uc?export=view&id=1lKPgaw-Ak4WK_91aBMcLvUMJL6pDQpgO)

### レジストリに新しいモデルがリンクされるときにSlack通知を設定

![model registry](https://drive.google.com/uc?export=view&id=1RsWCa6maJYD5y34gQ0nwWiKSWUCqcjT9)

## Registered Modelの利用
APIを介して対応する`name:alias`を参照することで、任意のRegistered Modelを利用できます。エンジニア、研究者、CI/CDプロセスのいずれであっても、テストを通過する必要があるかプロダクションに移行する必要があるすべてのモデルの集中ハブとしてモデルレジストリにアクセスできます。

```notebook
%%wandb -h 600

run = wandb.init(project=PROJECT_NAME, entity=ENTITY, job_type='inference')
artifact = run.use_artifact(f'{ENTITY}/model-registry/Model Registry Tutorial:staging', type='model')
artifact_dir = artifact.download()
wandb.finish()
```

# 次は何？
次のチュートリアルでは、大規模言語モデルを反復処理し、W&B Promptsを使用してデバッグする方法を学びます：

## 👉 [Iterate on LLMs](prompts)