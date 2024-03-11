---
slug: /guides/integrations
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 인테그레이션

Weights & Biases 인테그레이션은 기존 프로젝트 내에서 실험 추적 및 데이터 버전 관리를 빠르고 쉽게 설정할 수 있게 해줍니다. 인기 있는 ML 프레임워크([PyTorch](pytorch.md)), 라이브러리([Hugging Face](huggingface.md)) 또는 서비스([SageMaker](other/sagemaker.md))를 사용하고 있다면, 아래 인테그레이션과 왼쪽 네비게이션 바에서 확인하세요!

### 관련 링크

* [예제](https://github.com/wandb/examples): 각 인테그레이션에 대한 노트북 및 스크립트 예제로 코드를 시도해 보세요
* [동영상 튜토리얼](https://www.youtube.com/playlist?list=PLD80i8An1OEGajeVo15ohAQYF1Ttle0lk): YouTube 동영상 튜토리얼을 통해 W&B 사용 방법을 배우세요

<iframe width="668" height="376" src="https://www.youtube.com/embed/hmewPDNUNJs?list=PLD80i8An1OEGajeVo15ohAQYF1Ttle0lk" title="Log Your First Run With W&amp;B" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

## 특정 인테그레이션을 위한 가이드

<Tabs
  defaultValue="frameworks"
  values={[
    {label: '인기 있는 ML 프레임워크', value: 'frameworks'},
    {label: '인기 있는 ML 라이브러리', value: 'repositories'},
    {label: '인기 있는 툴', value: 'tools'},
  ]}>
  <TabItem value="frameworks">

* [Keras](keras.md)
* [PyTorch](pytorch.md)
* [PyTorch Lightning](lightning.md)
* [PyTorch Ignite](other/ignite.md)
* [TensorFlow](tensorflow.md)
* [Fastai](fastai/README.md)
* [Scikit-Learn](scikit.md)


  </TabItem>
  <TabItem value="repositories">

* [Hugging Face](huggingface.md)
* [PyTorch Geometric](pytorch-geometric.md)
* [spaCy](spacy.md)
* [Ultralytics](ultralytics.md)
* [YOLOv5](yolov5.md)
* [Simple Transformers](other/simpletransformers.md)
* [Catalyst](other/catalyst.md)
* [XGBoost](xgboost.md)
* [LightGBM](lightgbm.md)


  </TabItem>
  <TabItem value="tools">

* [TensorBoard](tensorboard.md)
* [SageMaker](other/sagemaker.md)
* [Kubeflow Pipelines](other/kubeflow-pipelines-kfp.md)
* [Dagster](./dagster.md)
* [Docker](other/docker.md)
* [Databricks](other/databricks.md)
* [Ray Tune](other/ray-tune.md)
* [OpenAI Gym](other/openai-gym.md)


  </TabItem>
</Tabs>