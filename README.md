# Edge-CV

A Python project that exports a pretrained **ResNet50** model to ONNX, applies graph optimizations and INT8 dynamic quantization, and benchmarks the performance improvements in terms of model size, latency, throughput, and output similarity.

## Features

- Export PyTorch ResNet50 to ONNX
- Graph optimization using ONNX Runtime
- Dynamic INT8 quantization
- Benchmark FP32, optimized, and INT8 models
- Compare model size, latency, FPS, and speedup
- Measure output similarity between FP32 and INT8 models

## Tech Stack

- Python
- PyTorch
- TorchVision
- ONNX
- ONNX Runtime
- NumPy

## Installation

```bash
git clone https://github.com/pranav2006/Edge-CV.git
cd Edge-CV
pip install -r requirements.txt
```

## Usage

Run the complete optimization and benchmarking pipeline:

```bash
python main.py
```

The script will:

1. Export ResNet50 to ONNX
2. Apply graph optimizations
3. Generate an INT8 quantized model
4. Benchmark all models
5. Display performance comparisons

## Output Metrics

- Model Size (MB)
- Average Latency (ms)
- Throughput (FPS)
- Speedup Factor
- FP32 vs INT8 Output Similarity

## Generated Files

```
models/
├── resnet50_base.onnx
├── resnet50_optimized.onnx
├── resnet50_preprocessed.onnx
└── resnet50_int8.onnx
```
