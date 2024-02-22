---
slug: /guides/integrations
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 통합

Weights & Biases 통합은 기존 프로젝트 내에서 실험 추적 및 데이터 버전 관리를 빠르고 쉽게 설정할 수 있게 해줍니다. 만약 인기 있는 ML 프레임워크([PyTorch](pytorch.md)), 라이브러리([Hugging Face](huggingface.md)), 또는 서비스([SageMaker](other/sagemaker.md))를 사용하고 있다면, 아래의 통합과 왼쪽 탐색 모음에서 확인하세요!

### 관련 링크

* [예제](https://github.com/wandb/examples): 각 통합에 대한 노트북 및 스크립트 예제로 코드를 시도해보세요
* [비디오 튜토리얼](https://www.youtube.com/playlist?list=PLD80i8An1OEGajeVo15ohAQYF1Ttle0lk): YouTube 비디오 튜토리얼로 W&B 사용법을 배워보세요

<iframe width="668" height="376" src="https://www.youtube.com/embed/hmewPDNUNJs?list=PLD80i8An1OEGajeVo15ohAQYF1Ttle0lk" title="Log Your First Run With W&amp;B" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

## 특정 통합을 위한 가이드

<Tabs
  defaultValue="frameworks"
  values={[
    {label: '인기 있는 ML 프레임워크', value: 'frameworks'},
    {label: '인기 있는 ML 라이브러리', value: 'repositories'},
    {label: '인기 있는 도구', value: 'tools'},
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