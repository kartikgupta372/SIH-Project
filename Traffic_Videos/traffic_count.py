import cv2
import json
import os
from datetime import datetime
from ultralytics import YOLO

# -------------------------
# SETTINGS
# -------------------------
# Change to 0 for webcam (live feed)
video_source = "traffic1.mp4"   # or 0 for live feed

# Load YOLOv8 Nano (fastest on CPU)
model = YOLO("yolov8n.pt")

# Open video file or webcam
cap = cv2.VideoCapture(video_source)

# Store results for JSON
data_log = []

# Determine if it's live feed or video file
is_live = isinstance(video_source, int) or str(video_source) == "0"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference
    results = model(frame, verbose=False)

    count = 0
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            # Only count vehicles (car=2, motorbike=3, bus=5, truck=7)
            if cls in [2, 3, 5, 7]:
                count += 1

    # Show result on frame
    annotated = results[0].plot()
    cv2.putText(annotated, f"Vehicles: {count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Traffic Count", annotated)

    # -------------------------
    # Save data into log
    # -------------------------
    if is_live:
        # Get current time & date
        now = datetime.now()
        log_entry = {
            "mode": "live",
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "vehicles": count
        }
    else:
        # Get video timestamp
        frame_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # in seconds
        video_name = os.path.basename(video_source)
        log_entry = {
            "mode": "video",
            "video_name": video_name,
            "timestamp_sec": round(frame_time, 2),
            "vehicles": count
        }

    data_log.append(log_entry)

    # -------------------------
    # Press 'q' to stop early
    # -------------------------
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save all results to JSON
with open("traffic_counts.json", "w") as f:
    json.dump(data_log, f, indent=4)

print("✅ Done! Results saved to traffic_counts.json")
