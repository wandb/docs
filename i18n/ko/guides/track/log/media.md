---
description: Log rich media, from 3D point clouds and molecules to HTML and histograms
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 미디어 및 오브젝트 로깅

이미지, 비디오, 오디오 등을 지원합니다. 풍부한 미디어를 로깅하여 결과를 탐색하고 run, 모델, 데이터셋을 시각적으로 비교해보세요. 예시 및 사용 방법 가이드는 계속해서 읽어보세요.

:::info
미디어 타입에 대한 참조 문서를 찾고 있나요? [이 페이지](../../../ref/python/data-types/README.md)를 참조하세요.
:::

:::info
이러한 미디어 오브젝트를 로깅하는 작동 코드는 [이 Colab 노트북](http://wandb.me/media-colab)에서 확인할 수 있으며, wandb.ai에서 결과를 [여기](https://wandb.ai/lavanyashukla/visualize-predictions/reports/Visualize-Model-Predictions--Vmlldzo1NjM4OA)에서 확인하고, 위에 링크된 비디오 튜토리얼을 따라할 수 있습니다.
:::

## 이미지

입력, 출력, 필터 가중치, 활성화 등을 로깅하여 추적하세요!

![Inputs and outputs of an autoencoder network performing in-painting.](/images/track/log_images.png)

이미지는 NumPy 배열, PIL 이미지 또는 파일 시스템에서 직접 로깅할 수 있습니다.

:::info
트레이닝 중 로깅이 병목 현상이 되지 않도록 하고 결과를 볼 때 이미지 로딩이 병목 현상이 되지 않도록 스텝 당 50개 미만의 이미지를 로깅하는 것이 좋습니다.
:::

<Tabs
  defaultValue="arrays"
  values={[
    {label: '배열을 이미지로 로깅하기', value: 'arrays'},
    {label: 'PIL 이미지 로깅하기', value: 'pil_images'},
    {label: '파일에서 이미지 로깅하기', value: 'images_files'},
  ]}>
  <TabItem value="arrays">

예를 들어, [`torchvision`](https://pytorch.org/vision/stable/utils.html#torchvision.utils.make\_grid)에서 `make_grid`를 사용하여 수동으로 이미지를 구성할 때 배열을 직접 제공하세요.

배열은 [Pillow](https://pillow.readthedocs.io/en/stable/index.html)를 사용하여 png로 변환됩니다.

```python
images = wandb.Image(image_array, caption="Top: Output, Bottom: Input")

wandb.log({"examples": images})
```

이미지가 회색조인 경우 마지막 차원이 1이고, RGB인 경우 3이며, RGBA인 경우 4입니다. 배열에 부동 소수점이 포함된 경우 `0`에서 `255` 사이의 정수로 변환합니다. 이미지를 다르게 정규화하고 싶다면, [`mode`](https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes)를 수동으로 지정하거나 [`PIL.Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html)을 직접 제공하면 됩니다. 이 패널의 "PIL 이미지 로깅하기" 탭에서 설명한 것처럼요.
  </TabItem>
  <TabItem value="pil_images">

배열을 이미지로 변환하는 것에 대해 더 많은 제어를 원한다면, [`PIL.Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html)를 직접 구성하고 제공하세요.

```python
images = [PIL.Image.fromarray(image) for image in image_array]

wandb.log({"examples": [wandb.Image(image) for image in images]})
```
  </TabItem>
  <TabItem value="images_files">
더 많은 제어를 원한다면, 이미지를 원하는 대로 생성하고, 디스크에 저장하고, 파일 경로를 제공하세요.

```python
im = PIL.fromarray(...)
rgb_im = im.convert("RGB")
rgb_im.save("myimage.jpg")

wandb.log({"example": wandb.Image("myimage.jpg")})
```
  </TabItem>
</Tabs>

## 이미지 오버레이

<Tabs
  defaultValue="segmentation_masks"
  values={[
    {label: '분할 마스크', value: 'segmentation_masks'},
    {label: '경계 상자', value: 'bounding_boxes'},
  ]}>
  <TabItem value="segmentation_masks">

W&B UI에서 (불투명도 조정, 시간에 따른 변경 사항 보기 등) 상호 작용하며 시멘틱 세그멘테이션 마스크를 로깅하세요.

![Interactive mask viewing in the W&B UI.](/images/track/semantic_segmentation.gif)

오버레이를 로깅하려면 `wandb.Image`의 `masks` 키워드 인수에 다음 키와 값을 포함하는 사전을 제공해야 합니다:

* 이미지 마스크를 나타내는 두 가지 키 중 하나:
  * `"mask_data"`: 각 픽셀에 대한 정수 클래스 레이블을 포함하는 2D NumPy 배열
  * `"path"`: (문자열) 저장된 이미지 마스크 파일 경로
* `"class_labels"`: (선택 사항) 이미지 마스크의 정수 클래스 레이블을 읽을 수 있는 클래스 이름에 매핑하는 사전

여러 마스크를 로깅하려면 아래 코드 조각과 같이 여러 키를 포함하는 마스크 사전을 로깅하세요.

[실시간 예시 보기 →](https://app.wandb.ai/stacey/deep-drive/reports/Image-Masks-for-Semantic-Segmentation--Vmlldzo4MTUwMw)

[샘플 코드 →](https://colab.research.google.com/drive/1SOVl3EvW82Q4QKJXX6JtHye4wFix\_P4J)

```python
mask_data = np.array([[1, 2, 2, ..., 2, 2, 1], ...])

class_labels = {1: "tree", 2: "car", 3: "road"}

mask_img = wandb.Image(
    image,
    masks={
        "predictions": {"mask_data": mask_data, "class_labels": class_labels},
        "ground_truth": {
            # ...
        },
        # ...
    },
)
```
  </TabItem>
  <TabItem value="bounding_boxes">
이미지에 경계 상자를 로깅하고, 필터와 토글을 사용하여 UI에서 다른 상자 세트를 동적으로 시각화하세요.

![](@site/static/images/track/bb-docs.jpeg)

[실시간 예시 보기 →](https://app.wandb.ai/stacey/yolo-drive/reports/Bounding-Boxes-for-Object-Detection--Vmlldzo4Nzg4MQ)

경계 상자를 로깅하려면 `wandb.Image`의 boxes 키워드 인수에 다음 키와 값을 포함하는 사전을 제공해야 합니다:

* `box_data`: 각 상자에 대해 하나씩, 사전의 리스트입니다. 상자 사전 형식은 아래에 설명되어 있습니다.
  * `position`: 아래에 설명된 두 가지 형식 중 하나로 상자의 위치와 크기를 나타내는 사전입니다. 모든 상자가 동일한 형식을 사용할 필요는 없습니다.
    * _옵션 1:_ `{"minX", "maxX", "minY", "maxY"}`. 각 상자 차원의 상한과 하한을 정의하는 좌표 세트를 제공합니다.
    * _옵션 2:_ `{"middle", "width", "height"}`. `middle` 좌표를 `[x,y]`로, `width`와 `height`를 스칼라로 지정하는 좌표 세트를 제공합니다.
  * `class_id`: 상자의 클래스 신원을 나타내는 정수입니다. 아래 `class_labels` 키를 참조하세요.
  * `scores`: UI에서 상자를 필터링하는 데 사용될 수 있는 스코어에 대한 문자열 레이블과 숫자 값의 사전입니다.
  * `domain`: 상자 좌표의 단위/형식을 지정합니다. 상자 좌표가 픽셀 공간에서 표현되는 경우(즉, 이미지 차원의 경계 내에서 정수로 표현됨) **이를 "pixel"로 설정하세요**. 기본적으로 도메인은 이미지의 분수/백분율(0과 1 사이의 부동 소수점)로 가정됩니다.
  * `box_caption`: (선택 사항) 이 상자에 표시될 레이블 텍스트로 사용될 문자열입니다.
* `class_labels`: (선택 사항) `class_id`를 문자열에 매핑하는 사전입니다. 기본적으로 `class_0`, `class_1` 등의 클래스 레이블을 생성합니다.

이 예제를 확인하세요:

```python
class_id_to_label = {
    1: "car",
    2: "road",
    3: "building",
    # ...
}

img = wandb.Image(
    image,
    boxes={
        "predictions": {
            "box_data": [
                {
                    # one box expressed in the default relative/fractional domain
                    "position": {"minX": 0.1, "maxX": 0.2, "minY": 0.3, "maxY": 0.4},
                    "class_id": 2,
                    "box_caption": class_id_to_label[2],
                    "scores": {"acc": 0.1, "loss": 1.2},
                    # another box expressed in the pixel domain
                    # (for illustration purposes only, all boxes are likely
                    # to be in the same domain/format)
                    "position": {"middle": [150, 20], "width": 68, "height": 112},
                    "domain": "pixel",
                    "class_id": 3,
                    "box_caption": "a building",
                    "scores": {"acc": 0.5, "loss": 0.7},
                    # ...
                    # Log as many boxes an as needed
                }
            ],
            "class_labels": class_id_to_label,
        },
        # Log each meaningful group of boxes with a unique key name
        "ground_truth": {
            # ...
        },
    },
)

wandb.log({"driving_scene": img})
```
  </TabItem>
</Tabs>

## 테이블에서의 이미지 오버레이

<Tabs
  defaultValue="segmentation_masks"
  values={[
    {label: '분할 마스크', value: 'segmentation_masks'},
    {label: '경계 상자', value: 'bounding_boxes'},
  ]}>
  <TabItem value="segmentation_masks">

![Interactive Segmentation Masks in Tables](/images/track/Segmentation_Masks.gif)

테이블에서 분할 마스크를 로깅하려면, 각 행에 대해 `wandb.Image` 오브젝트를 제공해야 합니다.

아래 코드 조각에서 예시가 제공됩니다:

```python
table = wandb.Table(columns=["ID", "Image"])

for id, img, label in zip(ids, images, labels):
    mask_img = wandb.Image(
        img,
        masks={
            "prediction": {"mask_data": label, "class_labels": class_labels}
            # ...
        },
    )

    table.add_data(id, img)

wandb.log({"Table": table})
```
  </TabItem>
  <TabItem value="bounding_boxes">


![Interactive Bounding Boxes in Tables](/images/track/Bounding_Boxes.gif)

테이블에서 경계 상자가 있는 이미지를 로깅하려면, 각 행에 대해 `wandb.Image` 오브젝트를 제공해야 합니다.

아래 코드 조각에서 예시가 제공됩니다:

```python
table = wandb.Table(columns=["ID", "Image"])

for id, img, boxes in zip(ids, images, boxes_set):
    box_img = wandb.Image(
        img,
        boxes={
            "prediction": {
                "box_data": [
                    {
                        "position": {
                            "minX": box["minX"],
                            "minY": box["minY"],
                            "maxX": box["maxX"],
                            "maxY": box["maxY"],
                        },
                        "class_id": box["class_id"],
                        "box_caption": box["caption"],
                        "domain": "pixel",
                    }
                    for box in boxes
                ],
                "class_labels": class_labels,
            }
        },
    )
```
  </TabItem>
</Tabs>

## 히스토그램

<Tabs
  defaultValue="histogram_logging"
  values={[
    {label: '기본 히스토그램 로깅', value: 'histogram_logging'},
    {label: '유연한 히스토그램 로깅', value: 'flexible_histogram'},
    {label: '요약에서의 히스토그램', value: 'histogram_summary'},
  ]}>
  <TabItem value="histogram_logging">
  
첫 번째 인수로 숫자 시퀀스(예: 리스트, 배열, 텐서)가 제공되면, `np.histogram`을 호출하여 히스토그램을 자동으로 구성합니다. 모든 배열/텐서는 평탄화됩니다. 기본값인 `64`개의 bin을 변경하려면 선택적 키워드 인수 `num_bins`를 사용할 수 있습니다. 지원되는 최대 bin 수는 `512`입니다.

UI에서는 훈련 단계를 x축에, 메트릭 값을 y축에 두고, 색상으로 카운트를 나타내어 훈련 중에 로깅된 히스토그램을 비교하기 쉽게 히스토그램을 그립니다. 이 패널의 "요약에서의 히스토그램" 탭에서 일회성 히스토그램 로깅에 대한 세부 정보를 확인하세요.

```python
wandb.log({"gradients": wandb.Histogram(grads)})
```

![Gradients for the discriminator in a GAN.](/images/track/histograms.png)
  </TabItem>
  <TabItem value="flexible_histogram">

더 많은 제어를 원한다면, `np.histogram`을 호출하고 반환된 튜플을 `np_histogram` 키워드 인수로 전달하세요.

```python
np_hist_grads = np.histogram(grads, density=True, range=(0.0, 1.0))
wandb.log({"gradients": wandb.Histogram(np_hist_grads)})
```
  </TabItem>
  <TabItem value="histogram_summary">

```python
wandb.run.summary.update(  # 요약에만 있는 경우, Overview 탭에서만 보임
    {"final_logits": wandb.Histogram(logits)}
)
```
  </TabItem>
</Tabs>

히스토그램이 요약에 있는 경우 [Run Page](../../app/pages/run-page.md)의 Overview 탭에 표시됩니다. 히스토그램이 기록에 있는 경우, 차트 탭에서 시간에 따른 bin의 열지도를 그립니다.

## 3D 시각화

<Tabs
  defaultValue="3d_object"
  values={[
    {label: '3D 오브젝트', value: '3d_object'},
    {label: '포인트 클라우드', value: 'point_clouds'},
    {label: '분자', value: 'molecules'},
  ]}>
  <TabItem value="3d_object">

`'obj', 'gltf', 'glb', 'babylon', 'stl', 'pts.json'` 형식의 파일을 로깅하면, run이 완료될 때 UI에서 이를 렌더링합니다.

```python
wandb.log(
    {
        "generated_samples": [
            wandb.Object3D(open("sample.obj")),
            wandb.Object3D(open("sample.gltf")),
            wandb.Object3D(open("sample.glb")),
        ]
    }
)
```

![Ground truth and prediction of a headphones point cloud](/images/track/ground_truth_prediction_of_3d_point_clouds.png)

[

### 이미지나 미디어를 업로드하지 않고 W&B를 내 프로젝트에 통합하고 싶다면 어떻게 해야 하나요?

W&B는 스칼라만 로그하는 프로젝트에서도 사용할 수 있으며 업로드하고 싶은 파일이나 데이터를 명시적으로 지정할 수 있습니다. 이미지를 로그하지 않는 [PyTorch에서의 빠른 예제](http://wandb.me/pytorch-colab)가 있습니다.

### PNG를 로그하는 방법은?

[`wandb.Image`](../../../ref/python/data-types/image.md)는 `numpy` 배열이나 `PILImage` 인스턴스를 기본적으로 PNG로 변환합니다.

```python
wandb.log({"example": wandb.Image(...)})
# 혹은 여러 이미지
wandb.log({"example": [wandb.Image(...) for img in images]})
```

### 비디오를 로그하는 방법은?

비디오는 [`wandb.Video`](../../../ref/python/data-types/video.md) 데이터 타입을 사용하여 로그됩니다:

```python
wandb.log({"example": wandb.Video("myvideo.mp4")})
```

이제 미디어 브라우저에서 비디오를 볼 수 있습니다. 프로젝트 워크스페이스, run 워크스페이스, 또는 리포트로 이동하여 "시각화 추가"를 클릭하여 리치 미디어 패널을 추가하세요.

### 포인트 클라우드에서 탐색하고 확대하는 방법은?

컨트롤을 누른 상태에서 마우스를 사용하여 공간 안에서 이동할 수 있습니다.

### 분자의 2D 뷰를 로그하는 방법은?

[`wandb.Image`](../../../ref/python/data-types/image.md) 데이터 타입과 [`rdkit`](https://www.rdkit.org/docs/index.html)을 사용하여 분자의 2D 뷰를 로그할 수 있습니다:

```python
molecule = rdkit.Chem.MolFromSmiles("CC(=O)O")
rdkit.Chem.AllChem.Compute2DCoords(molecule)
rdkit.Chem.AllChem.GenerateDepictionMatching2DStructure(molecule, molecule)
pil_image = rdkit.Chem.Draw.MolToImage(molecule, size=(300, 300))

wandb.log({"acetic_acid": wandb.Image(pil_image)})
```