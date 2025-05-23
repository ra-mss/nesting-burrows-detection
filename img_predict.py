import os
from ultralytics import YOLO
import cv2

# Ruta del modelo previamente entrenado
model_path = '/Users/USER/Desktop/nidos_model/last.pt'

# Cargar el modelo YOLO
model = YOLO(model_path)

# Definir un umbral de confianza para las detecciones
threshold = 0.5

# Ruta de la carpeta que contiene las imágenes a evaluar
image_path = '/Users/USER/Desktop/nidos_model/images/test'

# Obtener la lista de archivos de imagen en la carpeta
image_files = [f for f in os.listdir(image_path)]

# Procesar cada imagen en la lista
for image_file in image_files:
    # Cargar la imagen
    image = cv2.imread(os.path.join(image_path, image_file))
    
    # Realizar la detección de objetos en la imagen
    results = model(image)
    
    # Obtener las detecciones y puntuaciones
    detections = results.pred[0].cpu().numpy()
    
    # Iterar sobre las detecciones y dibujar cajas y etiquetas
    for det in detections:
        x1, y1, x2, y2, score, class_id = det.tolist()
        
        # Filtrar detecciones por el umbral de confianza
        if score > threshold:
            # Dibujar caja delimitadora
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Mostrar etiqueta y puntuación
            label = f"{model.names[int(class_id)]}: {score:.2f}"
            cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Mostrar la imagen con las detecciones
    cv2.imshow('Detecciones', image)
    cv2.waitKey(0)

# Cerrar todas las ventanas de imagen al finalizar
cv2.destroyAllWindows()
