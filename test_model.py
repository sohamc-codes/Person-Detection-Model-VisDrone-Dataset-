# test_model_local.py
from ultralytics import YOLO
import cv2
from pathlib import Path

# Load model
model_path = 'runs/detect/models/detect/visdrone_forest_threat_gpu4/weights/best.pt'
model = YOLO(model_path)

class_names = ['person', 'vehicle', 'two-wheeler', 'other']

print("✅ Model loaded successfully")
print(f"   Model: {model_path}")
print(f"   Classes: {class_names}")
print("="*60)

# Get test images from validation set
val_images_dir = Path('data/val/images')
test_images = list(val_images_dir.glob('*.jpg'))[:10]  # Test first 10 images

# Create results directory
results_dir = Path('test_results')
results_dir.mkdir(exist_ok=True)

# Test each image
for img_path in test_images:
    print(f"\n{'='*60}")
    print(f"📸 Testing: {img_path.name}")
    
    # Run detection
    results = model(str(img_path), imgsz=1280, conf=0.25, device=0)  # device=0 for GPU
    
    # Parse results
    detections = results[0].boxes
    print(f"   Found {len(detections)} objects:")
    
    # Count by class
    class_counts = {name: 0 for name in class_names}
    for box in detections:
        cls_id = int(box.cls)
        conf = float(box.conf)
        class_counts[class_names[cls_id]] += 1
        
        # Get box coordinates (xyxy format)
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        print(f"     • {class_names[cls_id]}: {conf:.3f} confidence at [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")
    
    print(f"   Summary: {class_counts}")
    
    # Save annotated image
    annotated = results[0].plot()
    save_path = results_dir / f"{img_path.stem}_detected.jpg"
    cv2.imwrite(str(save_path), annotated)
    print(f"   💾 Saved: {save_path}")

print("\n" + "="*60)
print(f"✅ Testing complete!")
print(f"📂 Results saved in: {results_dir.absolute()}")
print(f"📊 Tested {len(test_images)} images")

# Summary statistics
print("\n" + "="*60)
print("MODEL PERFORMANCE (from training):")
print("   mAP50:     0.659 (66% detection accuracy)")
print("   mAP50-95:  0.368 (37% across IoU 50-95%)")
print("   Precision: 0.714 (71% correct detections)")
print("   Recall:    0.609 (61% objects found)")