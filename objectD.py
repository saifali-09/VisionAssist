import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Using pre-trained YOLOv8 nano model

def detect_objects(frame):
    """
    Perform object detection on a given frame.
    """
    results = model(frame)
    return results

def draw_detections(frame, results):
    """
    Draw bounding boxes and labels on the frame.
    """
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            label = f"{model.names[cls]}: {conf:.2f}"
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def main():
    """
    Main function to capture webcam feed and run object detection.
    """
    cap = cv2.VideoCapture(1)  # Open webcam
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    while True:
        ret, frame = cap.read()
        # frame=cv2.flip(frame,1)
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        results = detect_objects(frame)
        frame = draw_detections(frame, results)
        
        cv2.imshow("YOLOv8 Real-Time Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
