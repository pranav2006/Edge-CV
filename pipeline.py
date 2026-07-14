import os
import time
import torch
import torchvision.models as models
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configuration block
MODEL_DIR = "models"
BASE_ONNX_PATH = os.path.join(MODEL_DIR, "resnet50_base.onnx")
OPT_ONNX_PATH = os.path.join(MODEL_DIR, "resnet50_optimized.onnx")
PREPROCESSED_ONNX_PATH = os.path.join(MODEL_DIR, "resnet50_preprocessed.onnx")
INT8_ONNX_PATH = os.path.join(MODEL_DIR, "resnet50_int8.onnx")
INPUT_SHAPE = (1, 3, 224, 224)
INPUT_NAME = "input_tensor"

# Ensure output directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# =====================================================================
# PHASE 1: Export Pre-trained Model to ONNX Base Format
# =====================================================================
def export_base_onnx():
    print("\n--- Phase 1: Exporting ResNet50 to Base ONNX ---")
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.eval()

    dummy_input = torch.randn(*INPUT_SHAPE)

    # 1. Export the initial graph structure
    torch.onnx.export(
        model,
        dummy_input,
        BASE_ONNX_PATH,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=[INPUT_NAME],
        output_names=["output_tensor"]
    )
    
    # 2. FORCE EMBED THE WEIGHTS: Load it and re-save it self-contained
    # This pulls any external shard files and bakes them straight into the main binary
    base_model = onnx.load(BASE_ONNX_PATH)
    onnx.save(base_model, BASE_ONNX_PATH)
    
    print(f"Base FP32 model exported successfully to: {BASE_ONNX_PATH}")

# =====================================================================
# PHASE 2: Structural Graph Optimization (Fusing Nodes)
# =====================================================================
def optimize_onnx_graph():
    print("\n--- Phase 2: Structural Graph Optimization ---")
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.optimized_model_filepath = OPT_ONNX_PATH
    
    # Run a quick session to force ONNX to serialize the fused graph to disk
    _ = ort.InferenceSession(BASE_ONNX_PATH, sess_options, providers=["CPUExecutionProvider"])
    opt_model = onnx.load(OPT_ONNX_PATH)
    onnx.save(opt_model, OPT_ONNX_PATH)
    print(f"Graph fused & optimized model saved to: {OPT_ONNX_PATH}")

# =====================================================================
# PHASE 3: Pre-processing & Dynamic INT8 Quantization
# =====================================================================
def quantize_to_int8():
    print("\n--- Phase 3: ONNX Pre-processing & INT8 Dynamic Quantization ---")
    
    # Run preprocessing on the highly predictable BASE graph instead of the fused graph
    quant_pre_process(
        input_model_path=BASE_ONNX_PATH,
        output_model_path=PREPROCESSED_ONNX_PATH,
        skip_symbolic_shape_inference=True
    )
    
    # Quantize the properly typed, pre-processed model graph
    quantize_dynamic(
        model_input=PREPROCESSED_ONNX_PATH,
        model_output=INT8_ONNX_PATH,
        weight_type=QuantType.QUInt8
    )
    print(f"Quantized INT8 model successfully saved to: {INT8_ONNX_PATH}")

# =====================================================================
# PHASE 4: The Benchmarking Execution Suite
# =====================================================================
def benchmark_model(model_path, iterations=100, warmup=15):
    # Disable internal runtime adjustments to evaluate the serialized file structure as-is
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    
    session = ort.InferenceSession(model_path, sess_options, providers=["CPUExecutionProvider"])
    test_data = np.random.randn(*INPUT_SHAPE).astype(np.float32)
    
    # Warmup runs to spin up CPU execution kernels
    for _ in range(warmup):
        _ = session.run(None, {INPUT_NAME: test_data})
        
    latencies = []
    outputs = []
    
    for _ in range(iterations):
        start_time = time.perf_counter()
        res = session.run(None, {INPUT_NAME: test_data})
        latencies.append(time.perf_counter() - start_time)
        outputs.append(res[0])
        
    avg_latency_ms = np.mean(latencies) * 1000
    fps = 1.0 / np.mean(latencies)
    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    
    return avg_latency_ms, fps, file_size_mb, outputs[0]

def run_benchmarking_suite():
    print("\n--- Phase 4: Benchmarking Latency and Performance Metrics ---")
    
    # Benchmark stages
    base_lat, base_fps, base_size, base_out = benchmark_model(BASE_ONNX_PATH)
    opt_lat, opt_fps, opt_size, opt_out = benchmark_model(OPT_ONNX_PATH)
    int8_lat, int8_fps, int8_size, int8_out = benchmark_model(INT8_ONNX_PATH)
    
    # Calculate Cosine Similarity to capture model output alignment against original base
    base_flat = base_out.flatten()
    int8_flat = int8_out.flatten()
    cosine_sim = np.dot(base_flat, int8_flat) / (np.linalg.norm(base_flat) * np.linalg.norm(int8_flat))
    sim_percentage_loss = (1.0 - cosine_sim) * 100

    # Output Results Table
    print("\n" + "="*75)
    print(f"{'METRIC':<22}{'BASE (FP32)':<16}{'OPTIMIZED (FUSED)':<20}{'QUANTIZED (INT8)':<15}")
    print("="*75)
    print(f"{'Model Size':<22}{base_size:.2f} MB{opt_size:19.2f} MB{int8_size:18.2f} MB")
    print(f"{'Avg Latency':<22}{base_lat:.2f} ms{opt_lat:19.2f} ms{int8_lat:18.2f} ms")
    print(f"{'Throughput':<22}{base_fps:.1f} FPS{opt_fps:18.1f} FPS{int8_fps:17.1f} FPS")
    print(f"{'Speedup Factor':<22}{'1.0x':<16}{f'{base_lat/opt_lat:.2f}x':<20}{f'{base_lat/int8_lat:.2f}x':<15}")
    print("-"*75)
    print(f"Calculated Structural Output Variance (Base vs INT8): Loss of {sim_percentage_loss:.4f}%")
    print("="*75)

# =====================================================================
# Orchestrator Entrypoint
# =====================================================================
if __name__ == "__main__":
    export_base_onnx()
    optimize_onnx_graph()
    quantize_to_int8()
    run_benchmarking_suite()