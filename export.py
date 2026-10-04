# # export_model.py
# from ultralytics import YOLO

# # Load your best model
# best = YOLO('runs/detect/models/detect/visdrone_forest_threat_gpu4/weights/best.pt')

# print("Exporting to TFLite (quantized for Raspberry Pi)...")
# best.export(format='tflite', int8=True, data='visdrone_custom.yaml')

# print("Exporting to ONNX (for edge devices)...")
# best.export(format='onnx', opset=12, simplify=True)

# print("✅ Export complete!")
# print("TFLite: runs/detect/models/detect/visdrone_forest_threat_gpu4/weights/best_saved_model/best_int8.tflite")
# print("ONNX:   runs/detect/models/detect/visdrone_forest_threat_gpu4/weights/best.onnx")


# export.py
from ultralytics import YOLO

best = YOLO('runs/detect/models/detect/visdrone_forest_threat_gpu4/weights/best.pt')

print("Exporting to ONNX...")
best.export(format='onnx', opset=12, simplify=True)

print("✅ Export complete!")
print("ONNX: runs/detect/models/detect/visdrone_forest_threat_gpu4/weights/best.onnx")
print("\nTo deploy on Raspberry Pi:")
print("1. Install: pip install onnxruntime")
print("2. Use: model = onnxruntime.InferenceSession('best.onnx')")