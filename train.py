# from ultralytics import YOLO

# # Use medium model — nano misses small aerial targets
# model = YOLO('yolov8m.pt')

# model.train(
#     data='visdrone_custom.yaml',
#     epochs=150,
#     imgsz=1280,          # VisDrone standard — critical for small objects
#     batch=8,             # Lower due to larger image size; use -1 to auto-tune
#     name='visdrone_forest_threat_gpu',
#     project='models/detect',
#     device=0,
#     workers=8,
#     patience=30,         # Forest models need more time to converge
#     save_period=10,
#     amp=True,
#     plots=True,
#     val=True,

#     # Forest-specific augmentation
#     hsv_h=0.015,
#     hsv_s=0.7,
#     hsv_v=0.4,
#     degrees=15,
#     flipud=0.3,
#     mosaic=1.0,
#     copy_paste=0.3,

#     # Close-object detection
#     overlap_mask=True,
# )

# # Export BEST weights, not last
# best = YOLO('models/detect/visdrone_forest_threat_gpu/weights/best.pt')
# best.export(format='tflite', int8=True, data='visdrone_custom.yaml')  # Quantized for RPi
# best.export(format='onnx', opset=12, simplify=True)

# print("✅ Training complete — best.pt exported")

from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8m.pt')

    model.train(
        data='visdrone_custom.yaml',
        epochs=150,
        imgsz=1280,
        batch=4,
        name='visdrone_forest_threat_gpu',
        project='models/detect',
        device=0,
        workers=8,
        patience=30,
        save_period=10,
        amp=True,
        plots=True,
        val=True,

        # Class imbalance
        cls=1.5,

        # Forest augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15,
        flipud=0.3,
        mosaic=1.0,
        copy_paste=0.3,
        scale=0.5,

        # Small object detection
        overlap_mask=True,
    )

    # Export BEST weights
    best = YOLO('models/detect/visdrone_forest_threat_gpu/weights/best.pt')
    best.export(format='tflite', int8=True, data='visdrone_custom.yaml')
    best.export(format='onnx', opset=12, simplify=True)

    print("✅ Done — best.pt exported to tflite + onnx")