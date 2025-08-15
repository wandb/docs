---
title: 미디어 및 오브젝트 로그
description: 3D 포인트 클라우드와 분자부터 HTML, 히스토그램까지 다양한 리치 미디어를 로그하세요.
menu:
  default:
    identifier: ko-guides-models-track-log-media
    parent: log-objects-and-media
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Log_(Almost)_Anything_with_W%26B_Media.ipynb" >}}

이미지, 비디오, 오디오 등 다양한 미디어를 지원합니다. 풍부한 미디어를 로그로 남겨 결과를 탐색하고, Runs, Models, Datasets 를 시각적으로 비교하세요. 아래 예시와 가이드에서 자세한 사용법을 확인하실 수 있습니다.

{{% alert %}}
자세한 내용은 [데이터 타입 레퍼런스]({{< relref path="/ref/python/sdk/data-types/" lang="ko" >}})를 참고하세요.
{{% /alert %}}

{{% alert %}}
더 많은 정보가 궁금하다면 [모델 예측값 시각화 데모 리포트](https://wandb.ai/lavanyashukla/visualize-predictions/reports/Visualize-Model-Predictions--Vmlldzo1NjM4OA)를 확인하거나 [동영상 튜토리얼](https://www.youtube.com/watch?v=96MxRvx15Ts)을 시청해 보세요.
{{% /alert %}}

## 사전 준비 사항
W&B SDK로 미디어 오브젝트를 로그하려면 추가 의존성 설치가 필요할 수 있습니다.
아래 명령어를 실행해 패키지를 설치할 수 있습니다:

```bash
pip install wandb[media]
```

## 이미지

이미지를 로그하여 입력값, 출력값, 필터 웨이트, 활성화 값 등 다양한 정보를 추적하세요.

{{< img src="/images/track/log_images.png" alt="Autoencoder inputs and outputs" >}}

이미지는 NumPy 배열, PIL 이미지, 파일 등 다양한 방식으로 직접 로그할 수 있습니다.

각 step에서 이미지를 로그할 때마다 UI에 저장되어 보여집니다. 이미지 패널을 확장하고 step 슬라이더로 다른 step의 이미지를 비교해보세요. 트레이닝 중 모델의 출력이 어떻게 변화하는지 한눈에 확인할 수 있습니다.

{{% alert %}}
트레이닝 및 결과 확인 시 병목 현상을 막기 위해, step 당 50개 미만의 이미지를 로그하는 것이 권장됩니다.
{{% /alert %}}

{{< tabpane text=true >}}
   {{% tab header="배열을 이미지로 로그하기" %}}
배열을 직접 전달해 수동으로 이미지를 생성할 수 있습니다. 예를 들어 [`torchvision`의 `make_grid`](https://pytorch.org/vision/stable/utils.html#torchvision.utils.make_grid) 등을 사용할 수 있습니다.

배열은 [Pillow](https://pillow.readthedocs.io/en/stable/index.html)를 활용해 png로 변환됩니다.

```python
import wandb

with wandb.init(project="image-log-example") as run:

    images = wandb.Image(image_array, caption="Top: Output, Bottom: Input")

    run.log({"examples": images})
```

마지막 차원이 1이면 그레이스케일, 3이면 RGB, 4면 RGBA로 이미지 채널을 자동으로 감지합니다. 배열의 값이 float이면 0~255의 정수로 변환합니다. 다른 방식으로 이미지를 정규화하고 싶거나 모드를 직접 지정하려면 [`mode`](https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes)를 지정하거나, 아래 "PIL 이미지 로그하기" 탭처럼 [`PIL.Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html)를 바로 전달할 수도 있습니다.
   {{% /tab %}}
   {{% tab header="PIL 이미지 로그하기" %}}
배열을 이미지로 변환하는 과정을 완전히 제어하고 싶을 때, [`PIL.Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html)를 직접 생성 후 로그할 수 있습니다.

```python
from PIL import Image

with wandb.init(project="") as run:
    # NumPy 배열로부터 PIL 이미지 생성
    image = Image.fromarray(image_array)

    # 필요 시 RGB로 변환
    if image.mode != "RGB":
        image = image.convert("RGB")

    # 이미지 로그
    run.log({"example": wandb.Image(image, caption="My Image")})
```

   {{% /tab %}}
   {{% tab header="파일로부터 이미지 로그" %}}
이미지를 원하는 방식으로 생성하여 디스크에 저장한 뒤, 파일 경로만 전달하면 됩니다.

```python
import wandb
from PIL import Image

with wandb.init(project="") as run:

    im = Image.fromarray(...)
    rgb_im = im.convert("RGB")
    rgb_im.save("myimage.jpg")

    run.log({"example": wandb.Image("myimage.jpg")})
```   
   {{% /tab %}}
{{< /tabpane >}}


## 이미지 오버레이


{{< tabpane text=true >}}
   {{% tab header="시멘틱 세그멘테이션 마스크" %}}
시멘틱 세그멘테이션 마스크를 로그하고, W&B UI에서 투명도 조절, 시간에 따른 변화 등 다양한 상호작용이 가능합니다.

{{< img src="/images/track/semantic_segmentation.gif" alt="Interactive mask viewing" >}}

오버레이를 로그하려면 `wandb.Image`의 `masks` 인자에 아래와 같은 키와 값을 가진 딕셔너리를 전달하세요:

* 이미지 마스크를 나타내는 키 중 하나:
  * `"mask_data"`: 각 픽셀의 클래스 레이블이 담긴 2차원 NumPy 배열
  * `"path"`: 저장된 이미지 마스크 파일의 경로(문자열)
* `"class_labels"`: (선택사항) 이미지 마스크 내 각 정수 레이블에 대한 사람이 읽을 수 있는 클래스명 매핑 딕셔너리

여러 개의 마스크를 로그하려면, 아래 코드조각처럼 여러 키를 가지는 마스크 딕셔너리를 사용하세요.

[실제 예시 보기](https://app.wandb.ai/stacey/deep-drive/reports/Image-Masks-for-Semantic-Segmentation--Vmlldzo4MTUwMw)

[샘플 코드](https://colab.research.google.com/drive/1SOVl3EvW82Q4QKJXX6JtHye4wFix_P4J)

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

각 키에 대한 세그멘테이션 마스크는 step 단위(매번 `run.log()` 호출)로 정의됩니다.  
- 같은 step에서 동일한 mask 키에 서로 다른 값을 제공하면, 가장 최근 값만 이미지에 적용됩니다.
- step마다 마스크 키가 다르면, 각 키의 값은 모두 보여지지만 시점별로 정의된 값만 이미지에 적용합니다. step에서 정의되지 않은 마스크의 가시성을 토글해도 이미지는 변하지 않습니다.
   {{% /tab %}}
    {{% tab header="바운딩 박스" %}}
이미지와 함께 바운딩 박스를 로그해, UI에서 다양한 필터와 토글을 사용해 박스 그룹을 비교할 수 있습니다.

{{< img src="/images/track/bb-docs.jpeg" alt="Bounding box example" >}}

[실제 예시 보기](https://app.wandb.ai/stacey/yolo-drive/reports/Bounding-Boxes-for-Object-Detection--Vmlldzo4Nzg4MQ)

바운딩 박스를 로그하려면, `boxes` 인수에 아래와 같은 딕셔너리를 전달해야 합니다:

* `box_data`: 각 박스를 위한 딕셔너리의 리스트, 아래 포맷 설명 참고
  * `position`: 박스의 위치와 크기를 나타내는 딕셔너리로, 아래 두 형식 중 하나 사용
    * _옵션 1:_ `{"minX", "maxX", "minY", "maxY"}` 각 차원의 상/하한 좌표
    * _옵션 2:_ `{"middle", "width", "height"}` `middle`은 `[x, y]`, `width`, `height`는 스칼라 값
  * `class_id`: 박스의 클래스 ID(정수). 아래의 `class_labels`로 매핑
  * `scores`: string-숫자 쌍의 딕셔너리, UI에서 박스 필터링에 사용 가능
  * `domain`: 박스 좌표의 단위/포맷 지정. **좌표가 픽셀 스페이스(이미지 내 정수)이면 "pixel"** 로 지정. 기본값은 전체 이미지의 비율(0~1 사이 float)
  * `box_caption`: (선택) 박스 라벨로 표시되는 문자열
* `class_labels`: (선택) 각 `class_id`를 string으로 매핑하는 딕셔너리. 기본값은 `class_0`, `class_1`, 등으로 생성

아래는 예시입니다:

```python
import wandb

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
                    # 기본 상대/비율 단위로 표현된 박스
                    "position": {"minX": 0.1, "maxX": 0.2, "minY": 0.3, "maxY": 0.4},
                    "class_id": 2,
                    "box_caption": class_id_to_label[2],
                    "scores": {"acc": 0.1, "loss": 1.2},
                    # 픽셀 단위로 표현된 또 다른 박스
                    # (예시용, 실제로는 모든 박스가 동일 단위 사용하는 것이 일반적)
                    "position": {"middle": [150, 20], "width": 68, "height": 112},
                    "domain": "pixel",
                    "class_id": 3,
                    "box_caption": "a building",
                    "scores": {"acc": 0.5, "loss": 0.7},
                    # ...
                    # 필요한 만큼 박스 추가
                }
            ],
            "class_labels": class_id_to_label,
        },
        # 의미 있는 박스 그룹 별로 unique 키로 로그
        "ground_truth": {
            # ...
        },
    },
)

with wandb.init(project="my_project") as run:
    run.log({"driving_scene": img})
```    
    {{% /tab %}}
{{< /tabpane >}}



## Tables 에서의 이미지 오버레이

{{< tabpane text=true >}}
   {{% tab header="시멘틱 세그멘테이션 마스크" %}}
{{< img src="/images/track/Segmentation_Masks.gif" alt="Interactive Segmentation Masks in Tables" >}}

Tables 에 시멘틱 세그멘테이션 마스크를 로그하려면, 각 행에 대해 `wandb.Image` 오브젝트를 제공해야 합니다.

코드 예시는 아래와 같습니다:

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

    table.add_data(id, mask_img)

with wandb.init(project="my_project") as run:
    run.log({"Table": table})
```   
   {{% /tab %}}
   {{% tab header="바운딩 박스" %}}
{{< img src="/images/track/Bounding_Boxes.gif" alt="Interactive Bounding Boxes in Tables" >}}

Tables 에 바운딩 박스가 있는 이미지를 로그하려면, 각 행에 대해 `wandb.Image` 오브젝트를 사용해야 합니다.

코드 예시는 아래와 같습니다:

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
   {{% /tab %}}
{{< /tabpane >}}



## 히스토그램

{{< tabpane text=true >}}
   {{% tab header="기본 히스토그램 로그" %}}
숫자의 시퀀스(리스트, 배열, 텐서 등)를 첫 번째 인자로 전달하면, 자동으로 `np.histogram`을 호출해 히스토그램을 생성합니다. 모든 배열/텐서는 flatten 처리됩니다. 기본 bin 수는 64이며, `num_bins` 키워드 인수로 변경할 수 있습니다. 최대 bin 수는 512입니다.

UI에서는 트레이닝 step이 x축, 지표 값이 y축, count가 색상으로 표시되어 트레이닝 동안의 분포 변화를 쉽게 비교할 수 있습니다. 단일 히스토그램을 로그하는 방법은 이 패널의 "Summary에서의 히스토그램" 탭을 참고하세요.

```python
run.log({"gradients": wandb.Histogram(grads)})
```

{{< img src="/images/track/histograms.png" alt="GAN discriminator gradients" >}}   
   {{% /tab %}}
   {{% tab header="유연한 히스토그램 로그" %}}
더 정교하게 로그하려면, 직접 `np.histogram`을 호출하고 결과 튜플을 `np_histogram` 인자에 전달합니다.

```python
np_hist_grads = np.histogram(grads, density=True, range=(0.0, 1.0))
run.log({"gradients": wandb.Histogram(np_hist_grads)})
```
   {{% /tab %}}
{{< /tabpane >}}



히스토그램이 summary 에 포함되어 있으면 [Run Page]({{< relref path="/guides/models/track/runs/" lang="ko" >}})의 Overview 탭에서 볼 수 있고, 히스토리에 있으면 Charts 탭에서 시간 흐름에 따른 히트맵으로 시각화됩니다.

## 3D 시각화

3D 포인트 클라우드와 라이다(Lidar) 장면을 바운딩 박스와 함께 로그할 수 있습니다. 랜더링할 포인트의 좌표, 색상 정보를 담은 NumPy 배열을 전달하면 됩니다.

```python
point_cloud = np.array([[0, 0, 0, COLOR]])

run.log({"point_cloud": wandb.Object3D(point_cloud)})
```

{{% alert %}}
W&B UI에서는 30만 개 포인트까지만 데이터를 표시합니다.
{{% /alert %}}

#### NumPy 배열 포맷

다양한 컬러 스킴을 위해 세 가지 NumPy 배열 포맷을 지원합니다.

* `[[x, y, z], ...]` `nx3`
* `[[x, y, z, c], ...]` `nx4`  `| c는 [1, 14] 범위 카테고리` (세그멘테이션에 유용)
* `[[x, y, z, r, g, b], ...]` `nx6 | r,g,b는 [0,255] (RGB)` 빨강, 초록, 파랑 채널

#### 파이썬 오브젝트

해당 스키마를 사용하면 파이썬 오브젝트로 정의한 뒤 [ `from_point_cloud` 메소드]({{< relref path="/ref/python/sdk/data-types/Object3D/#from_point_cloud" lang="ko" >}})에 전달할 수 있습니다.

* `points` 는 시각화할 포인트의 좌표와 색상을 담은 NumPy 배열(위 포맷 참고)
* `boxes` 는 다음 세 가지 속성이 있는 딕셔너리의 NumPy 배열:
  * `corners` - 8개의 꼭짓점 좌표
  * `label` - 박스에 표시될 라벨 문자열 (선택)
  * `color` - 박스 색상의 rgb 값
  * `score` - 박스에 표시되는 숫자(예: score>0.75인 박스만 필터링 표시). (선택)
* `type` 은 시각화할 씬 타입 문자열. 현재는 `"lidar/beta"`만 지원

```python
point_list = [
    [
        2566.571924017235, # x
        746.7817289698219, # y
        -15.269245470863748,# z
        76.5, # red
        127.5, # green
        89.46617199365393 # blue
    ],
    [ 2566.592983606823, 746.6791987335685, -15.275803826279521, 76.5, 127.5, 89.45471117247024 ],
    [ 2566.616361739416, 746.4903185513501, -15.28628929674075, 76.5, 127.5, 89.41336375503832 ],
    [ 2561.706014951675, 744.5349468458361, -14.877496818222781, 76.5, 127.5, 82.21868245418283 ],
    [ 2561.5281847916694, 744.2546118233013, -14.867862032341005, 76.5, 127.5, 81.87824684536432 ],
    [ 2561.3693562897465, 744.1804761656741, -14.854129178142523, 76.5, 127.5, 81.64137897587152 ],
    [ 2561.6093071504515, 744.0287526628543, -14.882135189841177, 76.5, 127.5, 81.89871499537098 ],
    # ... 이하 생략
]

run.log({"my_first_point_cloud": wandb.Object3D.from_point_cloud(
     points = point_list,
     boxes = [{
         "corners": [
                [ 2601.2765123137915, 767.5669506323393, -17.816764802288663 ],
                [ 2599.7259021588347, 769.0082337923552, -17.816764802288663 ],
                [ 2599.7259021588347, 769.0082337923552, -19.66876480228866 ],
                [ 2601.2765123137915, 767.5669506323393, -19.66876480228866 ],
                [ 2604.8684867834395, 771.4313904894723, -17.816764802288663 ],
                [ 2603.3178766284827, 772.8726736494882, -17.816764802288663 ],
                [ 2603.3178766284827, 772.8726736494882, -19.66876480228866 ],
                [ 2604.8684867834395, 771.4313904894723, -19.66876480228866 ]
        ],
         "color": [0, 0, 255], # 바운딩 박스의 RGB 컬러
         "label": "car", # 박스에 표시되는 문자열
         "score": 0.6 # 박스에 표시되는 숫자
     }],
     vectors = [
        {"start": [0, 0, 0], "end": [0.1, 0.2, 0.5], "color": [255, 0, 0]}, # color는 선택
     ],
     point_cloud_type = "lidar/beta",
)})
```

포인트 클라우드는 ctrl 키와 마우스를 조작해 3D 공간 안에서 시점을 이동할 수 있습니다.

#### 포인트 클라우드 파일

[ `from_file` 메소드]({{< relref path="/ref/python/sdk/data-types/Object3D/#from_file" lang="ko" >}})를 이용해, 포인트 클라우드 데이터가 저장된 JSON 파일을 불러올 수 있습니다.

```python
run.log({"my_cloud_from_file": wandb.Object3D.from_file(
     "./my_point_cloud.pts.json"
)})
```

포인트 클라우드 데이터 포맷 예시는 아래와 같습니다.

```json
{
    "boxes": [
        {
            "color": [
                0,
                255,
                0
            ],
            "score": 0.35,
            "label": "My label",
            "corners": [
                [
                    2589.695869075582,
                    760.7400443552185,
                    -18.044831294622487
                ],
                [
                    2590.719039645323,
                    762.3871153874499,
                    -18.044831294622487
                ],
                [
                    2590.719039645323,
                    762.3871153874499,
                    -19.54083129462249
                ],
                [
                    2589.695869075582,
                    760.7400443552185,
                    -19.54083129462249
                ],
                [
                    2594.9666662674313,
                    757.4657929961453,
                    -18.044831294622487
                ],
                [
                    2595.9898368371723,
                    759.1128640283766,
                    -18.044831294622487
                ],
                [
                    2595.9898368371723,
                    759.1128640283766,
                    -19.54083129462249
                ],
                [
                    2594.9666662674313,
                    757.4657929961453,
                    -19.54083129462249
                ]
            ]
        }
    ],
    "points": [
        [
            2566.571924017235,
            746.7817289698219,
            -15.269245470863748,
            76.5,
            127.5,
            89.46617199365393
        ],
        [
            2566.592983606823,
            746.6791987335685,
            -15.275803826279521,
            76.5,
            127.5,
            89.45471117247024
        ],
        [
            2566.616361739416,
            746.4903185513501,
            -15.28628929674075,
            76.5,
            127.5,
            89.41336375503832
        ]
    ],
    "type": "lidar/beta"
}
```
#### NumPy 배열

[위에서 설명한 배열 포맷]({{< relref path="#numpy-array-formats" lang="ko" >}}) 그대로, [ `from_numpy` 메소드]({{< relref path="/ref/python/sdk/data-types/Object3D/#from_numpy" lang="ko" >}})에 NumPy 배열을 전달해 포인트 클라우드를 정의할 수 있습니다.

```python
run.log({"my_cloud_from_numpy_xyz": wandb.Object3D.from_numpy(
     np.array(  
        [
            [0.4, 1, 1.3], # x, y, z
            [1, 1, 1], 
            [1.2, 1, 1.2]
        ]
    )
)})
```
```python
run.log({"my_cloud_from_numpy_cat": wandb.Object3D.from_numpy(
     np.array(  
        [
            [0.4, 1, 1.3, 1], # x, y, z, 카테고리 
            [1, 1, 1, 1], 
            [1.2, 1, 1.2, 12], 
            [1.2, 1, 1.3, 12], 
            [1.2, 1, 1.4, 12], 
            [1.2, 1, 1.5, 12], 
            [1.2, 1, 1.6, 11], 
            [1.2, 1, 1.7, 11], 
        ]
    )
)})
```
```python
run.log({"my_cloud_from_numpy_rgb": wandb.Object3D.from_numpy(
     np.array(  
        [
            [0.4, 1, 1.3, 255, 0, 0], # x, y, z, r, g, b 
            [1, 1, 1, 0, 255, 0], 
            [1.2, 1, 1.3, 0, 255, 255],
            [1.2, 1, 1.4, 0, 255, 255],
            [1.2, 1, 1.5, 0, 0, 255],
            [1.2, 1, 1.1, 0, 0, 255],
            [1.2, 1, 0.9, 0, 0, 255],
        ]
    )
)})
```

  </TabItem>
  <TabItem value="molecules">

```python
run.log({"protein": wandb.Molecule("6lu7.pdb")})
```

아래 10가지 파일 타입으로 분자 데이터를 로그할 수 있습니다: `pdb`, `pqr`, `mmcif`, `mcif`, `cif`, `sdf`, `sd`, `gro`, `mol2`, `mmtf`.

W&B는 SMILES 문자열, [`rdkit`](https://www.rdkit.org/docs/index.html) `mol` 파일, `rdkit.Chem.rdchem.Mol` 오브젝트 등도 지원합니다.

```python
resveratrol = rdkit.Chem.MolFromSmiles("Oc1ccc(cc1)C=Cc1cc(O)cc(c1)O")

run.log(
    {
        "resveratrol": wandb.Molecule.from_rdkit(resveratrol),
        "green fluorescent protein": wandb.Molecule.from_rdkit("2b3p.mol"),
        "acetaminophen": wandb.Molecule.from_smiles("CC(=O)Nc1ccc(O)cc1"),
    }
)
```

Run 이 종료되면, UI에서 분자를 3D로 상호작용하며 탐색할 수 있습니다.

[AlphaFold 활용 라이브 예시 보기](https://wandb.me/alphafold-workspace)

{{< img src="/images/track/docs-molecule.png" alt="Molecule structure" >}}
  </TabItem>
</Tabs>

### PNG 이미지

[`wandb.Image`]({{< relref path="/ref/python/sdk/data-types/image.md" lang="ko" >}})는 기본적으로 `numpy` 배열이나 `PILImage` 인스턴스를 PNG로 변환합니다.

```python
run.log({"example": wandb.Image(...)})
# 여러 이미지 로그
run.log({"example": [wandb.Image(...) for img in images]})
```

### 비디오

비디오는 [`wandb.Video`]({{< relref path="/ref/python/sdk/data-types/Video" lang="ko" >}}) 데이터 타입으로 로그합니다:

```python
run.log({"example": wandb.Video("myvideo.mp4")})
```

Project workspace, run workspace, 또는 report 에서 **시각화 추가**를 클릭하면 미디어 패널에서 비디오를 볼 수 있습니다.

## 분자의 2D 시각화

[`wandb.Image`]({{< relref path="/ref/python/sdk/data-types/image.md" lang="ko" >}}) 데이터 타입과 [`rdkit`](https://www.rdkit.org/docs/index.html)을 이용해 분자의 2D 시각화도 로그할 수 있습니다.

```python
molecule = rdkit.Chem.MolFromSmiles("CC(=O)O")
rdkit.Chem.AllChem.Compute2DCoords(molecule)
rdkit.Chem.AllChem.GenerateDepictionMatching2DStructure(molecule, molecule)
pil_image = rdkit.Chem.Draw.MolToImage(molecule, size=(300, 300))

run.log({"acetic_acid": wandb.Image(pil_image)})
```


## 기타 미디어

W&B는 다양한 미디어 타입의 로그도 지원합니다.

### 오디오

```python
run.log({"whale songs": wandb.Audio(np_array, caption="OooOoo", sample_rate=32)})
```

step 당 최대 100개 오디오 클립까지 로그할 수 있습니다. 더 자세한 사용법은 [`audio-file`]({{< relref path="/ref/query-panel/audio-file.md" lang="ko" >}})을 참고하세요.

### 비디오

```python
run.log({"video": wandb.Video(numpy_array_or_path_to_video, fps=4, format="gif")})
```

numpy 배열이 주어지면, 순서대로 시간, 채널, 넓이, 높이 차원이 있다고 가정합니다. 기본 설정은 4fps gif 이미지 생성이며([`ffmpeg`](https://www.ffmpeg.org) 와 [`moviepy`](https://pypi.org/project/moviepy/) 파이썬 라이브러리 필요). 지원 포맷은 `"gif"`, `"mp4"`, `"webm"`, `"ogg"`입니다. 문자열을 넘기면 해당 파일이 존재하며 지원 포맷인지 확인 후 업로드합니다. BytesIO 오브젝트를 넘기면 해당 포맷 확장자를 가진 임시 파일이 생성됩니다.

W&B [Run]({{< relref path="/guides/models/track/runs/" lang="ko" >}}) 및 [Project]({{< relref path="/guides/models/track/project-page.md" lang="ko" >}}) 페이지의 Media 섹션에서 비디오를 확인할 수 있습니다.

더 자세한 사용법은 [`video-file`]({{< relref path="/ref/query-panel/video-file" lang="ko" >}})을 참고하세요.

### 텍스트

UI 테이블에 텍스트를 표시하려면 `wandb.Table`을 사용해 로그하세요. 기본 컬럼 헤더는 `["Input", "Output", "Expected"]`입니다. UI 성능을 위해 기본 최대 행 수는 10,000개입니다. 필요하다면 `wandb.Table.MAX_ROWS = {DESIRED_MAX}`로 조절할 수 있습니다.

```python
with wandb.init(project="my_project") as run:
    columns = ["Text", "Predicted Sentiment", "True Sentiment"]
    # 방법 1
    data = [["I love my phone", "1", "1"], ["My phone sucks", "0", "-1"]]
    table = wandb.Table(data=data, columns=columns)
    run.log({"examples": table})

    # 방법 2
    table = wandb.Table(columns=columns)
    table.add_data("I love my phone", "1", "1")
    table.add_data("My phone sucks", "0", "-1")
    run.log({"examples": table})
```

또한 pandas `DataFrame` 오브젝트도 바로 전달할 수 있습니다.

```python
table = wandb.Table(dataframe=my_dataframe)
```

더 자세한 내용은 [`string`]({{< relref path="/ref/query-panel/" lang="ko" >}})을 참고하세요.

### HTML

```python
run.log({"custom_file": wandb.Html(open("some.html"))})
run.log({"custom_string": wandb.Html('<a href="https://mysite">Link</a>')})
```

커스텀 HTML은 어떤 키로든 로그할 수 있으며, Run 페이지에서 별도의 HTML 패널로 노출됩니다. 기본적으로 스타일도 자동 적용되며, `inject=False`로 기본 스타일 적용을 끌 수 있습니다.

```python
run.log({"custom_file": wandb.Html(open("some.html"), inject=False)})
```

더 자세한 사용법은 [`html-file`]({{< relref path="/ref/query-panel/html-file" lang="ko" >}})을 참고하세요.