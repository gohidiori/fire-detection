import cv2
from ultralytics import YOLO
import os

def main():
    # Use trained custom model
    model_path = "best.pt"
    
    if os.path.exists(model_path):
        model = YOLO(model_path)
        print("Menggunakan model deteksi api kustom (best.pt)")
    else:
        model = YOLO('yolov8n.pt')
        print("Model custom tidak ditemukan. Menggunakan model YOLOv8 default (COCO dataset)")
        print("Untuk deteksi api akurat, latih model kustom dengan dataset api")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Tidak bisa membuka kamera")
        return
    
    # Set resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Tekan 'q' untuk keluar")
    print("Tekan 's' untuk screenshot")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Tidak bisa membaca frame")
            break
        
        # Run inference
        results = model(frame, stream=True, verbose=False, conf=0.4)
        
        # Process results
        fire_detected = False
        for r in results:
            annotated_frame = r.plot()
            
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = model.names[cls]
                    
                    # Check for fire-related classes
                    fire_keywords = ['fire', 'flame', 'smoke', 'api', 'asap']
                    if any(kw in class_name.lower() for kw in fire_keywords):
                        fire_detected = True
                        # Draw extra warning
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.putText(annotated_frame, "FIRE DETECTED!", (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        # Show fire warning on frame
        if fire_detected:
            cv2.putText(annotated_frame, "WARNING: FIRE DETECTED!", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            # Add red border
            cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1]-1, annotated_frame.shape[0]-1), (0, 0, 255), 5)
        
        # Show FPS
        cv2.imshow('Fire Detection System', annotated_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite('fire_detection_screenshot.jpg', annotated_frame)
            print("Screenshot disimpan!")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()