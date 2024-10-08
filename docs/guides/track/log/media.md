---
title: Log media and objects
description: 3D 포인트 클라우드와 분자부터 HTML 및 히스토그램까지 다양한 미디어 로그 작성하기
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx'

<CTAButtons colabLink='https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Log_(Almost)_Anything_with_W%26B_Media.ipynb'/>

이미지, 비디오, 오디오 등 다양한 미디어를 지원합니다. 풍부한 미디어를 로그하여 결과를 탐색하고, Runs, Models, 그리고 Datasets를 시각적으로 비교하세요. 예제와 사용 방법 가이드를 계속 읽어보세요.

:::info
우리 미디어 타입에 대한 참고 문서가 필요하신가요? [이 페이지](../../../ref/python/data-types/README.md)를 보세요.
:::

:::info
[wandb.ai에서 결과가 어떻게 보이는지를 확인](https://wandb.ai/lavanyashukla/visualize-predictions/reports/Visualize-Model-Predictions--Vmlldzo1NjM4OA)하고, [비디오 튜토리얼을 따라가보세요](https://www.youtube.com/watch?v=96MxRvx15Ts).
:::

## 사전 요구사항
W&B SDK로 미디어 오브젝트를 로그하려면 추가 의존성을 설치해야 할 수 있습니다.
다음 코맨드를 실행하여 이러한 의존성을 설치할 수 있습니다:

```bash
pip install wandb[media]
```

## 이미지

이미지를 로그하여 입력값, 출력값, 필터 가중치, 활성화값 등을 추적하세요!

![자동 인페인팅을 수행하는 자동 인코더 네트워크의 입력과 출력.](/images/track/log_images.png)

이미지는 NumPy 배열, PIL 이미지, 혹은 파일 시스템에서 직접 로그할 수 있습니다.

:::info
트레이닝 중 로그가 병목 현상이 되는 것을 방지하고 결과를 볼 때 이미지 로딩이 병목 현상이 되는 것을 방지하기 위해 각 단계별로 50개 미만의 이미지를 로그하는 것을 권장합니다.
:::

<Tabs
  defaultValue="arrays"
  values={[
    {label: '배열을 이미지로 로깅', value: 'arrays'},
    {label: 'PIL 이미지를 로깅', value: 'pil_images'},
    {label: '파일에서 이미지를 로깅', value: 'images_files'},
  ]}>
  <TabItem value="arrays">

이미지를 수동으로 생성할 때 직접 배열을 제공하세요. 예를 들어 [`torchvision`의 `make_grid`](https://pytorch.org/vision/stable/utils.html#torchvision.utils.make_grid)를 사용할 수 있습니다.

배열은 [Pillow](https://pillow.readthedocs.io/en/stable/index.html)를 사용하여 png로 변환됩니다.

```python
images = wandb.Image(image_array, caption="Top: Output, Bottom: Input")

wandb.log({"examples": images})
```

마지막 차원이 1이면 그레이 스케일, 3이면 RGB, 4이면 RGBA라고 가정합니다. 배열에 float 값이 포함되어 있다면, `0`에서 `255` 사이의 정수로 변환합니다. 이미지를 다른 방식으로 정규화하려면, `mode`(https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes)를 수동으로 지정하거나, "PIL 이미지를 로깅" 탭에서 설명된 대로 [`PIL.Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html)를 제공하세요.
  </TabItem>
  <TabItem value="pil_images">

배열을 이미지로 변환하는 과정을 완전히 제어하려면 [`PIL.Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html)를 직접 구성하고 이를 제공하세요.

```python
images = [PIL.Image.fromarray(image) for image in image_array]

wandb.log({"examples": [wandb.Image(image) for image in images]})
```
  </TabItem>
  <TabItem value="images_files">
더 많은 제어가 필요하다면, 원하는 방식으로 이미지를 생성하고 디스크에 저장한 후 파일 경로를 제공하세요.

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
    {label: 'Segmentation Masks', value: 'segmentation_masks'},
    {label: 'Bounding Boxes', value: 'bounding_boxes'},
  ]}>
  <TabItem value="segmentation_masks">

시멘틱 세그멘테이션 마스크를 로그하고, 이를 통해 투명도 변경, 시간 경과에 따른 변화 보기 등을 W&B UI를 통해 상호작용해보세요.

![W&B UI에서의 인터랙티브 마스크 보기.](/images/track/semantic_segmentation.gif)

오버레이를 로그하려면, `masks` 키워드 인수에 다음의 키와 값을 포함하는 사전을 `wandb.Image`에 제공해야 합니다:

* 이미지 마스크를 나타내는 두 키 중 하나:
  * `"mask_data"`: 각 픽셀에 대해 정수 클래스 레이블이 포함된 2D NumPy 배열
  * `"path"`: 저장된 이미지 마스크 파일의 경로를 나타내는 문자열
* `"class_labels"`: (선택적) 이미지 마스크의 정수 클래스 레이블을 읽기 가능한 클래스 이름으로 매핑하는 사전

여러 마스크를 로그하려면, 아래 코드조각처럼 여러 키를 가지는 마스크 사전을 로그하세요.

[실시간 예제 보기](https://app.wandb.ai/stacey/deep-drive/reports/Image-Masks-for-Semantic-Segmentation--Vmlldzo4MTUwMw)

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
  </TabItem>
  <TabItem value="bounding_boxes">
이미지와 함께 바운딩 박스를 로그하고 필터 및 토글을 사용하여 UI에서 다양한 박스 세트를 동적으로 시각화하세요.

![](/images/track/bb-docs.jpeg)

[실시간 예제 보기](https://app.wandb.ai/stacey/yolo-drive/reports/Bounding-Boxes-for-Object-Detection--Vmlldzo4Nzg4MQ)

바운딩 박스를 로그하려면, `wandb.Image`의 `boxes` 키워드 인수에 다음의 키와 값을 포함하는 사전을 제공해야 합니다:

* `box_data`: 각 박스에 대한 사전 목록. 박스 사전 형식은 아래에 설명되어 있습니다.
  * `position`: 아래에 설명된 두 가지 형식 중 하나로 박스의 위치와 크기를 나타내는 사전 표현. 모든 박스가 동일한 형식을 사용할 필요는 없습니다.
    * _옵션 1:_ `{"minX", "maxX", "minY", "maxY"}`. 각 박스 차원의 상하 경계를 정의하는 좌표 세트를 제공합니다.
    * _옵션 2:_ `{"middle", "width", "height"}`. `middle` 좌표를 `[x,y]`로, `width`와 `height`를 스칼라로 지정하는 좌표 세트를 제공합니다.
  * `class_id`: 박스의 클래스 정체성을 나타내는 정수. 아래의 `class_labels` 키를 참고하세요.
  * `scores`: 문자열 레이블과 점수에 대한 숫자 값을 갖는 사전. UI에서 박스를 필터링하는 데 사용될 수 있습니다.
  * `domain`: 박스 좌표의 단위/형식을 지정합니다. 박스 좌표가 픽셀 공간으로 표현되어 있다면 **"pixel"로 설정**하십시오 (이미지 차원의 경계 내에서 정수로 표현됩니다). 기본적으로 도메인은 이미지의 비율/백분율(0과 1 사이의 부동 소수점 숫자)로 가정됩니다.
  * `box_caption`: (선택적) 이 박스에 라벨 텍스트로 표시될 문자열
* `class_labels`: (선택적) `class_id`를 문자열로 매핑하는 사전. 기본적으로 `class_0`, `class_1` 등의 클래스 레이블을 생성합니다.

이 예제를 체크해보세요:

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
                    # 기본 상대/분수 도메인으로 표현된 박스 하나
                    "position": {"minX": 0.1, "maxX": 0.2, "minY": 0.3, "maxY": 0.4},
                    "class_id": 2,
                    "box_caption": class_id_to_label[2],
                    "scores": {"acc": 0.1, "loss": 1.2},
                    # 픽셀 도메인으로 표현된 다른 박스
                    # (일러스트레이션 목적, 모든 박스가 같은 도메인/형식일 가능성 큼)
                    "position": {"middle": [150, 20], "width": 68, "height": 112},
                    "domain": "pixel",
                    "class_id": 3,
                    "box_caption": "a building",
                    "scores": {"acc": 0.5, "loss": 0.7},
                    # ...
                    # 필요한 만큼의 박스를 로그
                }
            ],
            "class_labels": class_id_to_label,
        },
        # 중요한 각 박스 그룹을 고유한 키 이름으로 로그
        "ground_truth": {
            # ...
        },
    },
)

wandb.log({"driving_scene": img})
```
  </TabItem>
</Tabs>

## 테이블 내의 이미지 오버레이

<Tabs
  defaultValue="segmentation_masks"
  values={[
    {label: 'Segmentation Masks', value: 'segmentation_masks'},
    {label: 'Bounding Boxes', value: 'bounding_boxes'},
  ]}>
  <TabItem value="segmentation_masks">

![테이블 내의 인터랙티브 시멘틱 세그멘테이션 마스크](/images/track/Segmentation_Masks.gif)

테이블 내에서 Segmentation Masks를 로그하려면, 각 행에 대해 `wandb.Image` 오브젝트를 제공해야 합니다.

아래 코드조각에서 예를 제공했습니다:

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


![테이블 내의 인터랙티브 바운딩 박스](/images/track/Bounding_Boxes.gif)

테이블 내에서 바운딩 박스와 함께 이미지를 로그하려면, 각 행에 대해 `wandb.Image` 오브젝트를 제공해야 합니다.

아래 코드조각에서 예를 제공합니다:

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
    {label: '요약에 있는 히스토그램', value: 'histogram_summary'},
  ]}>
  <TabItem value="histogram_logging">
  
숫자 시퀀스(예: 리스트, 배열, 텐서)를 첫 번째 인수로 제공하면, `np.histogram`을 호출하여 히스토그램을 자동으로 구성합니다. 모든 배열/텐서는 평탄화된다는 점을 유의하세요. 기본 `64`개의 빈을 덮어쓰려면 `num_bins` 키워드 인수를 사용할 수 있습니다. 지원되는 최대 빈 수는 `512`입니다.

UI에서 히스토그램은 x축에 트레이닝 스텝, y축에 메트릭 값, 색상으로 나타나는 카운트로 히스토그램이 로그되는 과정을 쉽게 비교할 수 있도록 합니다. 패널의 "요약에 있는 히스토그램" 탭을 참조하여 한 번의 히스토그램 로깅에 대한 자세한 내용을 확인하세요.

```python
wandb.log({"gradients": wandb.Histogram(grads)})
```

![GAN에서 판별기의 그레이디언트](/images/track/histograms.png)
  </TabItem>
  <TabItem value="flexible_histogram">

더 많은 제어가 필요한 경우, `np.histogram`을 호출하고 반환된 튜플을 `np_histogram` 키워드 인수에 전달하세요.

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

히스토그램이 요약에 있으면 [Run 페이지](../../app/pages/run-page.md)의 Overview 탭에 표시됩니다. 기록에 히스토그램이 있는 경우, Charts 탭에서 시간이 지남에 따라 빈의 히트맵을 그립니다.

## 3D 시각화

<Tabs
  defaultValue="3d_object"
  values={[
    {label: '3D 오브젝트', value: '3d_object'},
    {label: '포인트 클라우드', value: 'point_clouds'},
    {label: '분자', value: 'molecules'},
  ]}>
  <TabItem value="3d_object">

파일 형식 `'obj', 'gltf', 'glb', 'babylon', 'stl', 'pts.json'`로 파일을 로그하면, run이 종료됐을 때 UI에서 이를 렌더합니다.

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

![헤드폰 포인트 클라우드의 그라운드 트루스와 예측값](/images/track/ground_truth_prediction_of_3d_point_clouds.png)

[실시간 예제 보기](https://app.wandb.ai/nbaryd/SparseConvNet-examples_3d_segmentation/reports/Point-Clouds--Vmlldzo4ODcyMA)
  </TabItem>
  <TabItem value="point_clouds">

3D 포인트 클라우드와 라이다 씬을 바운딩 박스와 함께 로깅합니다. 렌더할 포인트의 좌표와 색상을 포함하는 NumPy 배열을 전달하세요. UI에서는 포인트를 300,000개까지 제한합니다.

```python
point_cloud = np.array([[0, 0, 0, COLOR]])

wandb.log({"point_cloud": wandb.Object3D(point_cloud)})
```

유연한 색상 스킴을 위해 세 가지 다른 형태의 NumPy 배열이 지원됩니다.

* `[[x, y, z], ...]` `nx3`
* `[[x, y, z, c], ...]` `nx4` `| c는 시멘틱 세그멘테이션에 유용한 범주로 [1, 14] 범위 내 번호입니다.`
* `[[x, y, z, r, g, b], ...]` `nx6 | r,g,b는 빨강, 초록, 파랑 색상 채널의 [0,255] 범위 내 값입니다.`

아래는 로깅 코드 예제입니다:

* `points`는 위의 간단한 포인트 클라우드 렌더러로 나타낸 것과 동일한 형식의 NumPy 배열입니다.
* `boxes`는 세 가지 속성을 가진 파이썬 사전의 NumPy 배열입니다:
  * `corners` - 여덟 개의 모서리 리스트
  * `label` - 박스에 렌더링될 레이블을 나타내는 문자열 (선택적)
  * `color` - 박스의 색상을 나타내는 RGB 값
* `type`은 렌더링할 씬 유형을 나타내는 문자열입니다. 현재 지원되는 값은 `lidar/beta`입니다.

```python
# W&B에서 포인트와 박스를 로그합니다.
point_scene = wandb.Object3D(
    {
        "type": "lidar/beta",
        "points": np.array(  # 포인트 클라우드에 포인트 추가
            [[0.4, 1, 1.3], [1, 1, 1], [1.2, 1, 1.2]]
        ),
        "boxes": np.array(  # 3D 박스 그리기
            [
                {
                    "corners": [
                        [0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [1, 0, 0],
                        [1, 1, 0],
                        [0, 1, 1],
                        [1, 0, 1],
                        [1, 1, 1],
                    ],
                    "label": "Box",
                    "color": [123, 321, 111],
                },
                {
                    "corners": [
                        [0, 0, 0],
                        [0, 2, 0],
                        [0, 0, 2],
                        [2, 0, 0],
                        [2, 2, 0],
                        [0, 2, 2],
                        [2, 0, 2],
                        [2, 2, 2],
                    ],
                    "label": "Box-2",
                    "color": [111, 321, 0],
                },
            ]
        ),
        "vectors": np.array(  # 3D 벡터 추가
            [{"start": [0, 0, 0], "end": [0.1, 0.2, 0.5]}]
        ),
    }
)
wandb.log({"point_scene": point_scene})
```
  </TabItem>
  <TabItem value="molecules">

```python
wandb.log({"protein": wandb.Molecule("6lu7.pdb")})
```

분자 데이터를 10가지 파일 타입으로 로깅합니다:`pdb`, `pqr`, `mmcif`, `mcif`, `cif`, `sdf`, `sd`, `gro`, `mol2`, 또는 `mmtf.`

W&B는 또한 SMILES 문자열, [`rdkit`](https://www.rdkit.org/docs/index.html) `mol` 파일, `rdkit.Chem.rdchem.Mol` 객체에서 분자 데이터를 로깅하는 것을 지원합니다.

```python
resveratrol = rdkit.Chem.MolFromSmiles("Oc1ccc(cc1)C=Cc1cc(O)cc(c1)O")

wandb.log(
    {
        "resveratrol": wandb.Molecule.from_rdkit(resveratrol),
        "green fluorescent protein": wandb.Molecule.from_rdkit("2b3p.mol"),
        "acetaminophen": wandb.Molecule.from_smiles("CC(=O)Nc1ccc(O)cc1"),
    }
)
```

Run이 끝나면 UI에서 분자의 3D 시각화와 상호작용할 수 있습니다.

[AlphaFold를 사용한 실시간 예제 보기](http://wandb.me/alphafold-workspace)

![](/images/track/docs-molecule.png)
  </TabItem>
</Tabs>

## 기타 미디어

W&B는 다양한 다른 미디어 타입의 로깅 또한 지원합니다.

<Tabs
  defaultValue="audio"
  values={[
    {label: 'Audio', value: 'audio'},
    {label: 'Video', value: 'video'},
    {label: 'Text', value: 'text'},
    {label: 'HTML', value: 'html'},
  ]}>
  <TabItem value="audio">

```python
wandb.log({"whale songs": wandb.Audio(np_array, caption="OooOoo", sample_rate=32)})
```

각 단계마다 로그할 수 있는 오디오 클립의 최대 수는 100개입니다.

  </TabItem>
  <TabItem value="video">

```python
wandb.log({"video": wandb.Video(numpy_array_or_path_to_video, fps=4, format="gif")})
```

numpy 배열이 제공되면 차원 순서대로 시간, 채널, 너비, 높이로 간주합니다. 기본적으로 4 fps의 gif 이미지를 생성합니다([`ffmpeg`](https://www.ffmpeg.org)와 [`moviepy`](https://pypi.org/project/moviepy/) 파이썬 라이브러리가 numpy 오브젝트를 전달할 때 필요). 지원되는 포맷은 `"gif"`, `"mp4"`, `"webm"`, `"ogg"`입니다. `wandb.Video`에 문자열을 전달하면 파일이 존재하며 지원되는 형식인지 확인한 후 wandb에 업로드합니다. `BytesIO` 객체를 전달하면 지정한 포맷을 확장자로 갖는 임시 파일이 생성됩니다.

W&B [Run](../../app/pages/run-page.md)과 [Project](../../app/pages/project-page.md) 페이지에서는 미디어 섹션에서 비디오를 볼 수 있습니다.

  </TabItem>
  <TabItem value="text">

`wandb.Table`을 사용하여 UI에 표시할 텍스트를 테이블에 로그하세요. 기본적으로 열 헤더는 `["Input", "Output", "Expected"]`입니다. 최적의 UI 성능을 보장하기 위해 기본 최대 행 수는 10,000으로 설정되었습니다. 그러나 사용자는 `wandb.Table.MAX_ROWS = {DESIRED_MAX}`로 최대치를 명시적으로 설정할 수 있습니다.

```python
columns = ["Text", "Predicted Sentiment", "True Sentiment"]
# 방법 1
data = [["I love my phone", "1", "1"], ["My phone sucks", "0", "-1"]]
table = wandb.Table(data=data, columns=columns)
wandb.log({"examples": table})

# 방법 2
table = wandb.Table(columns=columns)
table.add_data("I love my phone", "1", "1")
table.add_data("My phone sucks", "0", "-1")
wandb.log({"examples": table})
```

pandas `DataFrame` 객체를 전달할 수도 있습니다.

```python
table = wandb.Table(dataframe=my_dataframe)
```
  </TabItem>
  <TabItem value="html">

```python
wandb.log({"custom_file": wandb.Html(open("some.html"))})
wandb.log({"custom_string": wandb.Html('<a href="https://mysite">Link</a>')})
```

런 페이지에서 HTML 패널을 제공하는 모든 키에 커스텀 HTML을 로그할 수 있습니다. 기본적으로 기본 스타일을 삽입하지만, `inject=False`를 전달하여 기본 스타일 삽입을 비활성화할 수 있습니다.

```python
wandb.log({"custom_file": wandb.Html(open("some.html"), inject=False)})
```

  </TabItem>
</Tabs>

## 자주 묻는 질문

### 에포크나 스텝 간에 이미지를 비교하려면 어떻게 해야 하나요?

스텝마다 이미지를 로그할 때마다 UI에 표시하기 위해 저장합니다. 이미지 패널을 확장하고 스텝 슬라이더를 사용하여 다양한 스텝의 이미지를 살펴보세요. 이를 통해 모델의 출력이 트레이닝 동안 어떻게 변화하는지를 쉽게 비교할 수 있습니다.

### 내 프로젝트에 W&B를 통합하고 싶지만, 이미지나 미디어를 업로드하고 싶지 않은 경우는 어떻게 해야 하나요?

W&B는 단순히 스칼라만 로그하는 프로젝트에서도 사용할 수 있습니다 — 여러분이 업로드하고자 하는 파일이나 데이터를 명시적으로 지정할 수 있습니다. 이 간단한 PyTorch 예제를 [참고하세요](http://wandb.me/pytorch-colab), 이미지 로그는 없습니다.

### PNG를 어떻게 로그하나요?

[`wandb.Image`](../../../ref/python/data-types/image.md)는 기본적으로 `numpy` 배열이나 `PILImage` 인스턴스를 PNG로 변환합니다.

```python
wandb.log({"example": wandb.Image(...)})
# 또는 여러 이미지
wandb.log({"example": [wandb.Image(...) for img in images]})
```

### 비디오를 어떻게 로그하나요?

비디오는 [`wandb.Video`](../../../ref/python/data-types/video.md) 데이터 타입을 사용하여 로그됩니다:

```python
wandb.log({"example": wandb.Video("myvideo.mp4")})
```

컨텐츠를 미디어 브라우저에서 이제 볼 수 있습니다. 프로젝트 워크스페이스, 런 워크스페이스 또는 리포트로 이동하고 "Add visualization"을 클릭하여 풍부한 미디어 패널을 추가하세요.

### 포인트 클라우드에서 내비게이션하고 줌 인하려면 어떻게 해야 하나요?

컨트롤 키를 누르고 마우스를 사용하여 공간을 탐색할 수 있습니다.

### 분자의 2D 뷰를 어떻게 로그하나요?

[`wandb.Image`](../../../ref/python/data-types/image.md) 데이터 타입과 [`rdkit`](https://www.rdkit.org/docs/index.html)을 사용하여 분자의 2D 뷰를 로그할 수 있습니다:

```python
molecule = rdkit.Chem.MolFromSmiles("CC(=O)O")
rdkit.Chem.AllChem.Compute2DCoords(molecule)
rdkit.Chem.AllChem.GenerateDepictionMatching2DStructure(molecule, molecule)
pil_image = rdkit.Chem.Draw.MolToImage(molecule, size=(300, 300))

wandb.log({"acetic_acid": wandb.Image(pil_image)})
```