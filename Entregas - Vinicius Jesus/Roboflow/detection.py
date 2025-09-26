import cv2
from ultralytics import YOLO

video_path = 'video/video.mp4'              
output_path = 'video/output_video.mp4'     
box_color = (255, 200, 100)  # (B, G, R)
box_thickness = 2
font_scale = 0.5
font_thickness = 1
font = cv2.FONT_HERSHEY_SIMPLEX

# Cores por classe
class_colors = {
    "mask": (0, 255, 0),       # Verde
    "no-mask": (0, 0, 255)     # Vermelho
}

model = YOLO('model/best.pt')
cap = cv2.VideoCapture(video_path)
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
        # Coordenadas da box
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Confiança e classe
        conf = float(box.conf[0])
        class_id = int(box.cls[0])
        label = model.names[class_id].lower()  # Garante que seja lowercase

        # Cor baseada na classe
        box_color = class_colors.get(label, (255, 255, 255))  # Branco se não for "mask"/"no-mask"

        # Desenhar bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, box_thickness)

        # Escrever o label com a confiança
        text = f"{label} {conf:.2f}"
        cv2.putText(frame, text, (x1, y1 - 10), font, font_scale, box_color, font_thickness, lineType=cv2.LINE_AA)

    # Exibir o frame com as detecções
    cv2.imshow("Detecção YOLO - Pressione ESC para sair", frame)

    # Salvar o frame no vídeo de saída
    out.write(frame)

    # Diminui a velocidade da exibição, mas não do vídeo salvo
    if cv2.waitKey(60) & 0xFF == 27:  # ESC para sair
        break

# Liberar recursos
cap.release()
out.release()
cv2.destroyAllWindows()