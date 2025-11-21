import os
from onnxruntime.quantization import quantize_dynamic, QuantType

# Paths to your original models
original_models = [
    "models/det.onnx",
    "models/rec.onnx"
]

def quantize_model(model_path):
    if not os.path.exists(model_path):
        print(f"Skipping {model_path} (File not found)")
        return

    # Define output name (e.g., det.onnx -> det_quant.onnx)
    output_path = model_path.replace(".onnx", "_quant.onnx")
    
    print(f"Processing {model_path}...")
    
    # Quantize: Converts Float32 weights to UInt8 (4x smaller)
    quantize_dynamic(
        model_input=model_path,
        model_output=output_path,
        weight_type=QuantType.QUInt8
    )
    
    original_size = os.path.getsize(model_path) / (1024 * 1024)
    new_size = os.path.getsize(output_path) / (1024 * 1024)
    
    print(f"Done! Saved to {output_path}")
    print(f"Size reduced: {original_size:.2f} MB -> {new_size:.2f} MB\n")

if __name__ == "__main__":
    print("--- Starting Quantization ---")
    for model in original_models:
        quantize_model(model)
    print("Complete. Upload the new '_quant.onnx' files to your server.")