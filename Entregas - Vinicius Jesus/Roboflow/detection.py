import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture('video/video.mp4')   #  usar vídeo
# cap = cv2.VideoCapture(0)                  #  usar webcam

output_path = 'video/output_video.mp4'     
box_color = (255, 200, 100)  # (B, G, R)
box_thickness = 2
font_scale = 0.5
font_thickness = 1
font = cv2.FONT_HERSHEY_SIMPLEX

class_colors = {
    "mask": (0, 255, 0),       # Verde
    "no-mask": (0, 0, 255)     # Vermelho
}

model = YOLO('model/best.pt')

# Se for vídeo, configura saída
out = None
if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0:  # só grava se não for webcam
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break 

    results = model(frame, verbose=False)
    detections = results[0]

    for box in detections.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        class_id = int(box.cls[0])
        label = model.names[class_id].lower()

        box_color = class_colors.get(label, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, box_thickness)

        text = f"{label} {conf:.2f}"
        cv2.putText(frame, text, (x1, y1 - 10), font, font_scale, box_color, font_thickness, lineType=cv2.LINE_AA)

    cv2.imshow("Deteccao YOLO", frame)

    if out:
        out.write(frame)

    if cv2.waitKey(60) & 0xFF == 27:
        break

cap.release()
if out:
    out.release()
cv2.destroyAllWindows()

