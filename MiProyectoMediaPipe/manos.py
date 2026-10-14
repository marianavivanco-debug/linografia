
import cv2
import mediapipe as mp
import time

# ==================================
# 1. CONFIGURAR MEDIAPIPE
# ==================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

opciones = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2
)

# ==================================
# 2. ABRIR LA CAMARA DEL MAC
# ==================================

# Probamos la cámara 1 primero
camara = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)

if not camara.isOpened():
    camara.release()
    print("Camara 1 no disponible. Probando camara 0...")
    camara = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not camara.isOpened():
    print("No se pudo abrir ninguna camara.")
    exit()

print("Camara conectada correctamente")

# ==================================
# 3. DETECTAR LAS MANOS
# ==================================

with HandLandmarker.create_from_options(opciones) as detector:

    inicio = time.monotonic()
    ultimo_tiempo = -1

    while True:

        correcto, imagen = camara.read()

        if not correcto:
            print("No se pudo obtener la imagen")
            break

        # Mostrar imagen como espejo
        imagen = cv2.flip(imagen, 1)

        # Convertir imagen a RGB
        rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

        frame = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        # Tiempo para MediaPipe
        tiempo = int((time.monotonic() - inicio) * 1000)
        tiempo = max(tiempo, ultimo_tiempo + 1)
        ultimo_tiempo = tiempo

        # Detectar manos
        resultado = detector.detect_for_video(frame, tiempo)

        alto, ancho, _ = imagen.shape

        # ==================================
        # 4. DIBUJAR PUNTOS DE LAS MANOS
        # ==================================

        for mano in resultado.hand_landmarks:

            for punto in mano:

                x = int(punto.x * ancho)
                y = int(punto.y * alto)

                cv2.circle(
                    imagen,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

        # Mostrar cantidad de manos
        cantidad = len(resultado.hand_landmarks)

        cv2.putText(
            imagen,
            f"Manos detectadas: {cantidad}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        # ==================================
        # 5. MOSTRAR LA CAMARA
        # ==================================

        cv2.imshow(
            "MediaPipe - Deteccion de manos",
            imagen
        )

        # Presionar Q para salir
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# ==================================
# 6. CERRAR CAMARA
# ==================================

camara.release()
cv2.destroyAllWindows()
