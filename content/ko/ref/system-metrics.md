---
description: W&B에 의해 자동으로 로그되는 메트릭.
menu:
  reference:
    identifier: ko-system-metrics
    parent: reference
title: 시스템 메트릭
weight: 50
---

이 페이지는 W&B SDK에 의해 추적되는 시스템 메트릭에 대한 자세한 정보를 제공합니다.

{{% alert %}}
`wandb`는 15초마다 시스템 메트릭을 자동으로 로그합니다.
{{% /alert %}}

## CPU

### Process CPU Percent (CPU)
프로세스의 CPU 사용률 백분율로, 사용 가능한 CPU 수로 정규화됩니다.

W&B는 이 메트릭에 `cpu` 태그를 할당합니다.

### Process CPU Threads 
프로세스에서 사용되는 스레드 수입니다.

W&B는 이 메트릭에 `proc.cpu.threads` 태그를 할당합니다.

<!-- New section -->

## Disk

기본적으로 사용량 메트릭은 `/` 경로에서 수집됩니다. 모니터링할 경로를 구성하려면 다음 설정을 사용하세요:

```python
run = wandb.init(
    settings=wandb.Settings(
        x_stats_disk_paths=("/System/Volumes/Data", "/home", "/mnt/data"),
    ),
)
```

### Disk Usage Percent
지정된 경로에 대한 전체 시스템 디스크 사용률을 백분율로 나타냅니다.

W&B는 이 메트릭에 `disk.{path}.usagePercent` 태그를 할당합니다.

### Disk Usage
지정된 경로에 대한 전체 시스템 디스크 사용량을 기가바이트(GB)로 나타냅니다.
접근 가능한 경로가 샘플링되고, 각 경로의 디스크 사용량(GB 단위)이 샘플에 추가됩니다.

W&B는 이 메트릭에 `disk.{path}.usageGB` 태그를 할당합니다.

### Disk In
전체 시스템 디스크 읽기를 메가바이트(MB)로 나타냅니다.
첫 번째 샘플이 수집될 때 초기 디스크 읽기 바이트가 기록됩니다. 후속 샘플은 현재 읽기 바이트와 초기 값의 차이를 계산합니다.

W&B는 이 메트릭에 `disk.in` 태그를 할당합니다.

### Disk Out
전체 시스템 디스크 쓰기를 메가바이트(MB)로 나타냅니다.
[Disk In]({{< relref "#disk-in" >}})과 유사하게, 첫 번째 샘플이 수집될 때 초기 디스크 쓰기 바이트가 기록됩니다. 후속 샘플은 현재 쓰기 바이트와 초기 값의 차이를 계산합니다.

W&B는 이 메트릭에 `disk.out` 태그를 할당합니다.

<!-- New section -->

## Memory

### Process Memory RSS
프로세스의 메모리 Resident Set Size(RSS)를 메가바이트(MB)로 나타냅니다. RSS는 주 메모리(RAM)에 보관되는 프로세스가 점유하는 메모리 부분입니다.

W&B는 이 메트릭에 `proc.memory.rssMB` 태그를 할당합니다.

### Process Memory Percent
전체 사용 가능한 메모리에 대한 프로세스의 메모리 사용량을 백분율로 나타냅니다.

W&B는 이 메트릭에 `proc.memory.percent` 태그를 할당합니다.

### Memory Percent
전체 사용 가능한 메모리에 대한 전체 시스템 메모리 사용량을 백분율로 나타냅니다.

W&B는 이 메트릭에 `memory_percent` 태그를 할당합니다.

### Memory Available
사용 가능한 전체 시스템 메모리를 메가바이트(MB)로 나타냅니다.

W&B는 이 메트릭에 `proc.memory.availableMB` 태그를 할당합니다.

<!-- New section -->
## Network

### Network Sent
네트워크를 통해 전송된 총 바이트를 나타냅니다.
메트릭이 처음 초기화될 때 초기 전송 바이트가 기록됩니다. 후속 샘플은 현재 전송 바이트와 초기 값의 차이를 계산합니다.

W&B는 이 메트릭에 `network.sent` 태그를 할당합니다.

### Network Received

네트워크를 통해 수신된 총 바이트를 나타냅니다.
[Network Sent]({{< relref "#network-sent" >}})와 유사하게, 메트릭이 처음 초기화될 때 초기 수신 바이트가 기록됩니다. 후속 샘플은 현재 수신 바이트와 초기 값의 차이를 계산합니다.

W&B는 이 메트릭에 `network.recv` 태그를 할당합니다.

<!-- New section -->
## NVIDIA GPU

아래에 설명된 메트릭 외에도, 프로세스 및/또는 그 하위 항목이 특정 GPU를 사용하는 경우, W&B는 해당 메트릭을 `gpu.process.{gpu_index}.{metric_name}`으로 캡처합니다

### GPU Memory Utilization
각 GPU의 GPU 메모리 사용률을 백분율로 나타냅니다.

W&B는 이 메트릭에 `gpu.{gpu_index}.memory` 태그를 할당합니다.

### GPU Memory Allocated
각 GPU의 총 사용 가능한 메모리에 대한 GPU 메모리 할당량을 백분율로 나타냅니다.

W&B는 이 메트릭에 `gpu.{gpu_index}.memoryAllocated` 태그를 할당합니다.

### GPU Memory Allocated Bytes
각 GPU의 GPU 메모리 할당량을 바이트로 지정합니다.

W&B는 이 메트릭에 `gpu.{gpu_index}.memoryAllocatedBytes` 태그를 할당합니다.

### GPU Utilization
각 GPU의 GPU 사용률을 백분율로 반영합니다.

W&B는 이 메트릭에 `gpu.{gpu_index}.gpu` 태그를 할당합니다.

### GPU Temperature
각 GPU의 GPU 온도를 섭씨로 나타냅니다.

W&B는 이 메트릭에 `gpu.{gpu_index}.temp` 태그를 할당합니다.

### GPU Power Usage Watts
각 GPU의 GPU 전력 사용량을 와트로 나타냅니다.

W&B는 이 메트릭에 `gpu.{gpu_index}.powerWatts` 태그를 할당합니다.

### GPU Power Usage Percent

각 GPU의 전력 용량에 대한 GPU 전력 사용량의 백분율을 반영합니다.

W&B는 이 메트릭에 `gpu.{gpu_index}.powerPercent` 태그를 할당합니다.

### GPU SM Clock Speed 
GPU의 Streaming Multiprocessor(SM) 클록 속도를 MHz로 나타냅니다. 이 메트릭은 계산 작업을 담당하는 GPU 코어 내 처리 속도를 나타냅니다.

W&B는 이 메트릭에 `gpu.{gpu_index}.smClock` 태그를 할당합니다.

### GPU Memory Clock Speed
GPU 메모리의 클록 속도를 MHz로 나타냅니다. 이는 GPU 메모리와 처리 코어 간의 데이터 전송 속도에 영향을 줍니다.

W&B는 이 메트릭에 `gpu.{gpu_index}.memoryClock` 태그를 할당합니다.

### GPU Graphics Clock Speed 

GPU에서 그래픽 렌더링 작업을 위한 기본 클록 속도를 MHz로 나타냅니다. 이 메트릭은 시각화나 렌더링 작업 중 성능을 반영하는 경우가 많습니다.

W&B는 이 메트릭에 `gpu.{gpu_index}.graphicsClock` 태그를 할당합니다.

### GPU Corrected Memory Errors

W&B가 오류 검사 프로토콜에 의해 자동으로 수정하는 GPU의 메모리 오류 수를 추적하여 복구 가능한 하드웨어 문제를 나타냅니다.

W&B는 이 메트릭에 `gpu.{gpu_index}.correctedMemoryErrors` 태그를 할당합니다.

### GPU Uncorrected Memory Errors
W&B가 수정하지 않은 GPU의 메모리 오류 수를 추적하여 처리 신뢰성에 영향을 줄 수 있는 복구 불가능한 오류를 나타냅니다.

W&B는 이 메트릭에 `gpu.{gpu_index}.unCorrectedMemoryErrors` 태그를 할당합니다.

### GPU Encoder Utilization

GPU 비디오 인코더의 사용률을 백분율로 나타내며, 인코딩 작업(예: 비디오 렌더링)이 실행될 때의 부하를 나타냅니다.

W&B는 이 메트릭에 `gpu.{gpu_index}.encoderUtilization` 태그를 할당합니다.

<!-- New section -->
## AMD GPU
W&B는 AMD에서 제공하는 `rocm-smi` 도구(`rocm-smi -a --json`)의 출력에서 메트릭을 추출합니다.

ROCm [6.x(최신)](https://rocm.docs.amd.com/en/latest/) 및 [5.x](https://rocm.docs.amd.com/en/docs-5.6.0/) 형식이 지원됩니다. ROCm 형식에 대한 자세한 내용은 [AMD ROCm 문서](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)를 참조하세요. 새로운 형식에는 더 자세한 정보가 포함되어 있습니다.

### AMD GPU Utilization
각 AMD GPU 기기의 GPU 사용률을 백분율로 나타냅니다.

W&B는 이 메트릭에 `gpu.{gpu_index}.gpu` 태그를 할당합니다.

### AMD GPU Memory Allocated
각 AMD GPU 기기의 총 사용 가능한 메모리에 대한 GPU 메모리 할당량을 백분율로 나타냅니다.

W&B는 이 메트릭에 `gpu.{gpu_index}.memoryAllocated` 태그를 할당합니다.

### AMD GPU Temperature
각 AMD GPU 기기의 GPU 온도를 섭씨로 나타냅니다.

W&B는 이 메트릭에 `gpu.{gpu_index}.temp` 태그를 할당합니다.

### AMD GPU Power Usage Watts
각 AMD GPU 기기의 GPU 전력 사용량을 와트로 나타냅니다.

W&B는 이 메트릭에 `gpu.{gpu_index}.powerWatts` 태그를 할당합니다.

### AMD GPU Power Usage Percent
각 AMD GPU 기기의 전력 용량에 대한 GPU 전력 사용량의 백분율을 반영합니다.

W&B는 이 메트릭에 `gpu.{gpu_index}.powerPercent` 태그를 할당합니다.

<!-- New section -->
## Apple ARM Mac GPU

### Apple GPU Utilization
Apple GPU 기기, 특히 ARM Mac에서의 GPU 사용률을 백분율로 나타냅니다.

W&B는 이 메트릭에 `gpu.0.gpu` 태그를 할당합니다.

### Apple GPU Memory Allocated
ARM Mac의 Apple GPU 기기에서 총 사용 가능한 메모리에 대한 GPU 메모리 할당량을 백분율로 나타냅니다.

W&B는 이 메트릭에 `gpu.0.memoryAllocated` 태그를 할당합니다.

### Apple GPU Temperature
ARM Mac의 Apple GPU 기기의 GPU 온도를 섭씨로 나타냅니다.

W&B는 이 메트릭에 `gpu.0.temp` 태그를 할당합니다.

### Apple GPU Power Usage Watts
ARM Mac의 Apple GPU 기기의 GPU 전력 사용량을 와트로 나타냅니다.

W&B는 이 메트릭에 `gpu.0.powerWatts` 태그를 할당합니다.

### Apple GPU Power Usage Percent
ARM Mac의 Apple GPU 기기의 전력 용량에 대한 GPU 전력 사용량의 백분율을 나타냅니다.

W&B는 이 메트릭에 `gpu.0.powerPercent` 태그를 할당합니다.

<!-- New section -->
## Graphcore IPU
Graphcore IPU(Intelligence Processing Units)는 기계 지능 작업용으로 특별히 설계된 고유한 하드웨어 가속기입니다.

### IPU Device Metrics
이 메트릭은 특정 IPU 기기의 다양한 통계를 나타냅니다. 각 메트릭에는 이를 식별하는 기기 ID(`device_id`)와 메트릭 키(`metric_key`)가 있습니다. W&B는 이 메트릭에 `ipu.{device_id}.{metric_key}` 태그를 할당합니다.

메트릭은 Graphcore의 `gcipuinfo` 바이너리와 상호작용하는 독점 `gcipuinfo` 라이브러리를 사용하여 추출됩니다. `sample` 메서드는 프로세스 ID(`pid`)와 연결된 각 IPU 기기에 대해 이러한 메트릭을 가져옵니다. 시간이 지남에 따라 변경되는 메트릭이나 기기의 메트릭이 처음 가져올 때만 중복 데이터 로깅을 피하기 위해 로그됩니다.

각 메트릭에 대해 `parse_metric` 메서드를 사용하여 원시 문자열 표현에서 메트릭 값을 추출합니다. 그런 다음 메트릭은 `aggregate` 메서드를 사용하여 여러 샘플에 걸쳐 집계됩니다.

다음은 사용 가능한 메트릭과 해당 단위 목록입니다:

- **Average Board Temperature** (`average board temp (C)`): IPU 보드의 온도(섭씨).
- **Average Die Temperature** (`average die temp (C)`): IPU 다이의 온도(섭씨).
- **Clock Speed** (`clock (MHz)`): IPU의 클록 속도(MHz).
- **IPU Power** (`ipu power (W)`): IPU의 전력 소비(와트).
- **IPU Utilization** (`ipu utilisation (%)`): IPU 사용률 백분율.
- **IPU Session Utilization** (`ipu utilisation (session) (%)`): 현재 세션에 특정한 IPU 사용률 백분율.
- **Data Link Speed** (`speed (GT/s)`): 초당 기가 전송에서의 데이터 전송 속도.

<!-- New section -->

## Google Cloud TPU
Tensor Processing Unit(TPU)은 기계 학습 워크로드를 가속화하기 위한 Google의 맞춤 개발 ASIC(Application Specific Integrated Circuits)입니다.


### TPU Memory usage
TPU 코어당 현재 High Bandwidth Memory 사용량(바이트).

W&B는 이 메트릭에 `tpu.{tpu_index}.memoryUsageBytes` 태그를 할당합니다.

### TPU Memory usage percentage
TPU 코어당 현재 High Bandwidth Memory 사용량(백분율).

W&B는 이 메트릭에 `tpu.{tpu_index}.memoryUsageBytes` 태그를 할당합니다.

### TPU Duty cycle
TPU 기기당 TensorCore 듀티 사이클 백분율. 샘플 기간 동안 가속기 TensorCore가 적극적으로 처리한 시간의 백분율을 추적합니다. 값이 클수록 더 나은 TensorCore 활용을 의미합니다.

W&B는 이 메트릭에 `tpu.{tpu_index}.dutyCycle` 태그를 할당합니다.

<!-- New section -->

## AWS Trainium
[AWS Trainium](https://aws.amazon.com/machine-learning/trainium/)은 기계 학습 워크로드 가속화에 중점을 둔 AWS에서 제공하는 전문 하드웨어 플랫폼입니다. AWS의 `neuron-monitor` 도구가 AWS Trainium 메트릭을 캡처하는 데 사용됩니다.

### Trainium Neuron Core Utilization
각 NeuronCore의 사용률 백분율로, 코어별로 보고됩니다.

W&B는 이 메트릭에 `trn.{core_index}.neuroncore_utilization` 태그를 할당합니다.

### Trainium Host Memory Usage, Total 
호스트의 총 메모리 소비량(바이트).

W&B는 이 메트릭에 `trn.host_total_memory_usage` 태그를 할당합니다.

### Trainium Neuron Device Total Memory Usage 
Neuron 기기의 총 메모리 사용량(바이트).

W&B는 이 메트릭에 `trn.neuron_device_total_memory_usage)` 태그를 할당합니다.

### Trainium Host Memory Usage Breakdown:

다음은 호스트의 메모리 사용량 분석입니다:

- **Application Memory** (`trn.host_total_memory_usage.application_memory`): 애플리케이션에서 사용하는 메모리.
- **Constants** (`trn.host_total_memory_usage.constants`): 상수에 사용되는 메모리.
- **DMA Buffers** (`trn.host_total_memory_usage.dma_buffers`): Direct Memory Access 버퍼에 사용되는 메모리.
- **Tensors** (`trn.host_total_memory_usage.tensors`): 텐서에 사용되는 메모리.

### Trainium Neuron Core Memory Usage Breakdown
각 NeuronCore의 상세한 메모리 사용 정보:

- **Constants** (`trn.{core_index}.neuroncore_memory_usage.constants`)
- **Model Code** (`trn.{core_index}.neuroncore_memory_usage.model_code`)
- **Model Shared Scratchpad** (`trn.{core_index}.neuroncore_memory_usage.model_shared_scratchpad`)
- **Runtime Memory** (`trn.{core_index}.neuroncore_memory_usage.runtime_memory`)
- **Tensors** (`trn.{core_index}.neuroncore_memory_usage.tensors`)

## OpenMetrics
소비되는 엔드포인트에 적용될 맞춤 정규식 기반 메트릭 필터에 대한 지원과 함께 OpenMetrics / Prometheus 호환 데이터를 노출하는 외부 엔드포인트에서 메트릭을 캡처하고 로그합니다.

[NVIDIA DCGM-Exporter](https://docs.nvidia.com/datacenter/cloud-native/gpu-telemetry/latest/dcgm-exporter.html)를 사용한 GPU 클러스터 성능 모니터링의 특정 사례에서 이 기능을 사용하는 방법에 대한 자세한 예는 [W&B에서 GPU 클러스터 성능 모니터링](https://wandb.ai/dimaduev/dcgm/reports/Monitoring-GPU-cluster-performance-with-NVIDIA-DCGM-Exporter-and-Weights-Biases--Vmlldzo0MDYxMTA1)을 참조하세요.