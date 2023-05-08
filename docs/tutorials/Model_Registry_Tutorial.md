# Track Model Development Lifecycle 

<a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-model-registry/Model_Registry_E2E.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

![dataset_card_overview](https://drive.google.com/uc?export=view&id=1Ea_vEhhbcqHBnooY-Efdo_tST6ULRW3e)

1. Checkpoint the model every epoch and log as an artifact
2. Link your best model to a **Registered Collection** in the Model Registry
3. Retrieve the model from the collection for downstream evaluation or inference
4. Add aliases depending on stage of model

```python
import os
from random import shuffle
import numpy as np

# source directory for all raw data
DATA_SRC = "nature_100"
IMAGES_PER_LABEL = 10
BALANCED_SPLITS = {"train" : 8, "val" : 1, "test": 1}
SRC = DATA_SRC
PREFIX = "GCS" # convenient for tracking local data
PROJECT_NAME = "Model Registry E2E" #@param {type:"string"}
ENTITY="kenlee"#@param {type:"string"}
dataset_name = "mnist"

# number of images per class label
# the total number of images is 10X this (10 classes)
TOTAL_IMAGES = IMAGES_PER_LABEL * 10
RAW_DATA_AT = "_".join([PREFIX, "raw_data", str(TOTAL_IMAGES)])
```


```python
# set SIZE to "TINY", "SMALL", "MEDIUM", or "LARGE"
# to select one of these three datasets
# TINY dataset: 100 images, 30MB
# SMALL dataset: 1000 images, 312MB
# MEDIUM dataset: 5000 images, 1.5GB
# LARGE dataset: 12,000 images, 3.6GB

SIZE = "TINY"

if SIZE == "TINY":
  src_url = "https://storage.googleapis.com/wandb_datasets/nature_100.zip"
  src_zip = "nature_100.zip"
  DATA_SRC = "nature_100"
  IMAGES_PER_LABEL = 10
  BALANCED_SPLITS = {"train" : 8, "val" : 1, "test": 1}
elif SIZE == "SMALL":
  src_url = "https://storage.googleapis.com/wandb_datasets/nature_1K.zip"
  src_zip = "nature_1K.zip"
  DATA_SRC = "nature_1K"
  IMAGES_PER_LABEL = 100
  BALANCED_SPLITS = {"train" : 80, "val" : 10, "test": 10}
elif SIZE == "MEDIUM":
  src_url = "https://storage.googleapis.com/wandb_datasets/nature_12K.zip"
  src_zip = "nature_12K.zip"
  DATA_SRC = "inaturalist_12K/train" # (technically a subset of only 10K images)
  IMAGES_PER_LABEL = 500
  BALANCED_SPLITS = {"train" : 400, "val" : 50, "test": 50}
elif SIZE == "LARGE":
  src_url = "https://storage.googleapis.com/wandb_datasets/nature_12K.zip"
  src_zip = "nature_12K.zip"
  DATA_SRC = "inaturalist_12K/train" # (technically a subset of only 10K images)
  IMAGES_PER_LABEL = 1000
  BALANCED_SPLITS = {"train" : 800, "val" : 100, "test": 100}
```


```python
%%capture
!curl -SL $src_url > $src_zip
!unzip $src_zip
```

# Step 0: Setup

Start out by installing the experiment tracking library and setting up your free W&B account:


*   `pip install wandb` – Install the W&B library
*   `import wandb` – Import the wandb library
*   `wandb login` – Login to your W&B account so you can log all your metrics in one place


```python
!pip install -qqq wandb
import wandb
wandb.login()
```

# Step 1: Create a Registered Model!
![](https://drive.google.com/uc?id=13VM43j_7iaN8Hxn74yWWrn2nBismdvdU)

## Put the collection under the `model-registry` project in the team you want to make your model visible to:

![](https://drive.google.com/uc?id=1Y0xSfDktiC3l-OkBrUZmFh0v96d30eFM)


# Step 2: Log training data as an artifact
Check out more docs on [artifacts in W&B](https://docs.wandb.ai/guides/artifacts/api). Steps to create and log an artifact are quite simple
0. Initialize Run with `wandb.init()`
1. Create an artifact with `wandb.Artifact`
2. Add directories, files to the artifact with `artifact.add_dir`, `artifact.add_file`
3. Log the artifact with `wandb.log_artifact`


```python
!ls -l nature_100
```


```python
run = wandb.init(project=PROJECT_NAME, entity=ENTITY, job_type="upload_data")

raw_data_art = wandb.Artifact(RAW_DATA_AT, type="raw_data")
raw_data_art.add_dir(DATA_SRC)
run.log_artifact(raw_data_art)

run.finish()
```

# Log preprocessed/split data as artifact
- For example, a preprocessing job produces a tokenized or augmented dataset that is then utilized by a training job
- Each job is a run logged in W&B
- Declare dependency of a run on an artifact with `wandb.use_artifact`



```python
SPLIT_DATA_AT = "_".join([PREFIX, "80-10-10", str(TOTAL_IMAGES)])
run = wandb.init(project=PROJECT_NAME, entity=ENTITY, job_type="data_split")

SPLIT_COUNTS = BALANCED_SPLITS

data_at = run.use_artifact(RAW_DATA_AT + ":latest")
data_dir = data_at.download()
data_split_at = wandb.Artifact(SPLIT_DATA_AT, type="balanced_data")

labels = os.listdir(data_dir)
for l in labels:
  if l.startswith("."): # skip non-label file
    continue
  imgs_per_label = os.listdir(os.path.join(data_dir, l))
  shuffle(imgs_per_label)
  start_id = 0
  for split, count in SPLIT_COUNTS.items():
    # take a subset
    split_imgs = imgs_per_label[start_id:start_id+count]
    for img_file in split_imgs:
      f_id = img_file.split(".")[0]
      full_path = os.path.join(data_dir, l, img_file)

      data_split_at.add_file(full_path, name = os.path.join(split, l, img_file))
    start_id += count

# log artifact to W&B
run.log_artifact(data_split_at)
run.finish()
```


```python
# EXPERIMENT CONFIG
#------------------------
# Core globals to modify
NUM_EPOCHS = 5 # set low for demo purposes, try 3, or 5, or as many as you like


# optional globals to modify
# set to a custom name to help keep your experiments organized
RUN_NAME = "keras_model_training" 
# change this if you'd like start a new set of comparable Tables
# (only Tables logged to the same key can be compared)
VAL_TABLE_NAME = "predictions" 

# hyperparams set low for demo/training speed
# if you set these higher, be mindful of how many items are in
# the dataset artifacts you chose by setting the SIZE at the top
NUM_TRAIN = BALANCED_SPLITS["train"]*10
NUM_VAL = BALANCED_SPLITS["val"]*10

# enforced max for this is ceil(NUM_VAL/batch_size)
NUM_LOG_BATCHES = 16

# ARTIFACTS CONFIG
#------------------------
# training data artifact to load
TRAIN_DATA_AT = PREFIX + "_80-10-10_" + str(TOTAL_IMAGES)

# model name
# if you want to train a sufficiently different model, give this a new name
# to start a new lineage for the model, instead of just incrementing the
# version of the old model
MODEL_NAME = "iv3_finetuned"

# folder in which to save the final, trained model
# if you want to train a sufficiently different model, give this a new name
# to start a new lineage for the model, instead of just incrementing the
# version of the old model
SAVE_MODEL_DIR = "finetune_iv3_keras"

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from wandb.keras import WandbCallback

# experiment configuration saved to W&B
CFG = {
  "num_train" : NUM_TRAIN,
  "num_val" : NUM_VAL,
  "num_classes" : 10,
  "fc_size" : 1024,
  "epochs" : NUM_EPOCHS,
  "batch_size" : 32,

  # inceptionV3 settings
  "img_width" : 299,
  "img_height": 299
}

# number of validation data batches to log/use when computing metrics
# at the end of each epoch
max_log_batches = int(np.ceil(float(CFG["num_val"])/float(CFG["batch_size"])))
# change this min to max to log ALL the available images to a Table
CFG["num_log_batches"] = min(max_log_batches, NUM_LOG_BATCHES)

def finetune_inception_model(fc_size, num_classes):
  """Load InceptionV3 with ImageNet weights, freeze it,
  and attach a finetuning top for this classification task"""
  # load InceptionV3 as base
  base = InceptionV3(weights="imagenet", include_top="False")
  # freeze base layers
  for layer in base.layers:
    layer.trainable = False
  x = base.get_layer('mixed10').output 

  # attach a fine-tuning layer
  x = GlobalAveragePooling2D()(x)
  x = Dense(fc_size, activation='relu')(x)
  guesses = Dense(num_classes, activation='softmax')(x)

  model = Model(inputs=base.input, outputs=guesses)
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
  return model

def train():
  """ Main training loop which freezes the InceptionV3 layers of the model
  and only trains the new top layers on the new data. A subsequent training
  phase might unfreeze all the layers and finetune the whole model on the new data""" 
  run = wandb.init(project=PROJECT_NAME, entity=ENTITY, name=RUN_NAME, job_type="train", config=CFG)
  cfg = wandb.config

  # locate and download training and validation data
  data_at = TRAIN_DATA_AT + ":latest"
  data = run.use_artifact(data_at, type="balanced_data")
  data_dir = data.download()
  train_dir = os.path.join(data_dir, "train")
  val_dir = os.path.join(data_dir, "val")

  # create train and validation data generators
  train_datagen = ImageDataGenerator(
      rescale=1. / 255,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True)
  val_datagen = ImageDataGenerator(rescale=1. / 255)

  train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(cfg.img_width, cfg.img_height),
    batch_size=cfg.batch_size,
    class_mode='categorical')

  val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(cfg.img_width, cfg.img_height),
    batch_size=cfg.batch_size,
    class_mode='categorical',
    shuffle=False)

  # instantiate model and callbacks
  model = finetune_inception_model(cfg.fc_size, cfg.num_classes)
  callbacks = [WandbCallback(), ValLog(val_generator, cfg.num_log_batches)]

  # train!
  model.fit(
    train_generator,
    steps_per_epoch = cfg.num_train // cfg.batch_size,
    epochs=cfg.epochs,
    validation_data=val_generator,
    callbacks = callbacks,
    validation_steps = cfg.num_val // cfg.batch_size)

  
  run.finish()

```

# Step 3. Train and Checkpoint the Model
- Checkpoint the model every epoch and log as a model artifact
- Log metrics and predictions


```python

class ValLog(Callback):
  """ Custom callback to log validation images
  at the end of each training epoch"""
  def __init__(self, generator=None, num_log_batches=1):
    self.best_loss = float("inf")
    self.best_model = None

    self.generator = generator
    self.num_batches = num_log_batches
    # store full names of classes
    self.flat_class_names = [k for k, v in generator.class_indices.items()]

  def on_epoch_end(self, epoch, logs={}):
    # collect validation data and ground truth labels from generator
    val_data, val_labels = zip(*(self.generator[i] for i in range(self.num_batches)))
    val_data, val_labels = np.vstack(val_data), np.vstack(val_labels)

    # use the trained model to generate predictions for the given number
    # of validation data batches (num_batches)
    val_preds = self.model.predict(val_data)
    true_ids = val_labels.argmax(axis=1)
    max_preds = val_preds.argmax(axis=1)

    # log validation predictions alongside the run
    columns=["id", "image", "guess", "truth"]
    for a in self.flat_class_names:
      columns.append("score_" + a)
    predictions_table = wandb.Table(columns = columns)
    
    # log image, predicted and actual labels, and all scores
    for filepath, img, top_guess, scores, truth in zip(self.generator.filenames,
                                                       val_data, 
                                                       max_preds, 
                                                       val_preds,
                                                       true_ids):
      img_id = filepath.split('/')[-1].split(".")[0]
      row = [img_id, wandb.Image(img), 
             self.flat_class_names[top_guess], self.flat_class_names[truth]]
      for s in scores.tolist():
        row.append(np.round(s, 4))
      predictions_table.add_data(*row)

    val_acc = np.mean(max_preds == true_ids)
    wandb.run.log({VAL_TABLE_NAME : predictions_table,
                   'val_acc': val_acc})


    is_best = val_acc > self.best_loss
    if is_best:
        self.best_loss = val_acc
    
     # Checkpoint the Model at the end of each epoch
    trained_model_artifact = wandb.Artifact(
              MODEL_NAME, type="model",
              description="finetuned inception v3")
  
    self.model.save(SAVE_MODEL_DIR)
    trained_model_artifact.add_dir(SAVE_MODEL_DIR)

    # Add an alias indicating the best and latest checkpoint
    wandb.log_artifact(trained_model_artifact, aliases=["best", "latest"] if is_best else None)
    if is_best:
        self.best_model = trained_model_artifact
```


```python
train()
```

# Step 4. Link the best model checkpoint to the collection
1. You can link a model via the UI or api with [wandb.run.link_artifact](https://docs.wandb.ai/guides/models/walkthrough#3.-link-model-versions-to-the-collection)
2. Assign a `staging` alias to indicate this model is promising, but still needs further review


![](https://drive.google.com/uc?id=1fFdG_j0VZjCNsZ22hg-Gxfn08_Znw_JS)

# Step 5. Load your staged model from the collection for evaluation
- Perform evaluation and testing on the `staging` model. Refer to it by `Nature Classification:staging`




```python
TEST_TABLE_NAME = "test_results" 

from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
import os

MODEL_NAME = "iv3_finetuned"
# location of test data from our original split
# should match SPLIT_DATA_AT
TEST_DATA_AT = "_".join([PREFIX, "80-10-10", str(TOTAL_IMAGES)])


run = wandb.init(project=PROJECT_NAME, entity=ENTITY, job_type="inference")
model_at = wandb.use_artifact("Nature Classification:staging")
model_dir = model_at.download()
print("model: ", model_dir)
model = keras.models.load_model(model_dir)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# download latest version of test data
test_data_at = run.use_artifact(TEST_DATA_AT + ":latest")
test_dir = test_data_at.download()
test_dir += "/test/"

class_names = ["Animalia", "Amphibia", "Arachnida", "Aves", "Fungi", 
               "Insecta", "Mammalia", "Mollusca", "Plantae", "Reptilia"]

# load test images
imgs = []
filenames = []
class_labels = os.listdir(test_dir)
truth = []
for l in class_labels:
  if l.startswith("."):
    continue
  imgs_per_class = os.listdir(os.path.join(test_dir, l))
  for img in imgs_per_class:
    # track the image id
    filenames.append(img.split(".")[0])
    truth.append(l)
    img_path = os.path.join(test_dir, l, img)
    img = image.load_img(img_path, target_size=(299, 299))
    img = image.img_to_array(img)
    # don't forget to rescale test images to match the range of inputs
    # to the network
    img = np.expand_dims(img/255.0, axis=0)
    imgs.append(img)

# predict on test data and bin predictions by guessed label 
preds = {}
imgs = np.vstack(imgs)
classes = model.predict(imgs, batch_size=32)
for c in classes:
  class_id = np.argmax(c)
  if class_id in preds:
    preds[class_id] += 1
  else:
    preds[class_id] = 1

# log inference results as a Table to the run workspace
columns=["id", "image", "guess", "truth"]
for a in class_names:
  columns.append("score_" + a)
test_dt = wandb.Table(columns = columns)

# store all the scores for each image
for img_id, i, t, c in zip(filenames, imgs, truth, classes):
  guess = class_names[np.argmax(c)]
  row = [img_id, wandb.Image(i), guess, t]
  for c_i in c.tolist():
    row.append(np.round(c_i, 4))
  test_dt.add_data(*row)
  
run.log({TEST_TABLE_NAME : test_dt})
print("Quick distribution of predicted classes: ")
print(preds)
run.finish()
```

# Step 6. Replace Alias
- Replace `staging` with `production` alias on the model collection

![](https://drive.google.com/uc?id=1W5pRvTAqtjX30r8MZlc3eAkriQMebH8R)
