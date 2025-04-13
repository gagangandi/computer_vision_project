from ultralytics import YOLO
import cv2

def load_model():
    return YOLO('yolov5s.pt')  # You can use yolov8n.pt too

def detect_objects(model, image):
    results = model(image, verbose=False)[0]
    detections = []
    class_names = model.names

    annotated_image = image.copy()

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        width = x2 - x1
        class_id = int(box.cls[0])
        class_name = class_names[class_id]

        detections.append({
            "class_name": class_name,
            "width": float(width),
            "bbox": (x1, y1, x2, y2)
        })

        # Draw bounding box
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_image, class_name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return detections, annotated_image
