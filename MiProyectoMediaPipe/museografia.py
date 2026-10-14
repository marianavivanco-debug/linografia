
import cv2
import mediapipe as mp
import numpy as np
import time
import math
import os
from datetime import datetime

# ==========================================
# HUELLAS DE ORIGEN
# TALLER DE LINOGRAFIA INTERACTIVA
# ==========================================

W, H = 1200, 900
PAPER_BOX = (45, 380, 1155, 690)

PW = PAPER_BOX[2] - PAPER_BOX[0]
PH = PAPER_BOX[3] - PAPER_BOX[1]

# COLORES DE TINTA (BGR)
INKS = [
    ("TERRACOTA", (63, 92, 181)),
    ("INDIGO", (128, 80, 46)),
    ("OCRE", (62, 170, 205)),
    ("VERDE", (78, 115, 71)),
    ("NEGRO", (36, 38, 42)),
]

# MOTIVOS CULTURALES
# Son interpretaciones graficas educativas,
# no reproducciones de piezas originales.

MOTIFS = [
    (
        "Lukutuel",
        "Mapuche",
        "Centro-sur de Chile",
        "Figura textil interpretada en algunos "
        "contextos como una persona en actitud "
        "de rogativa."
    ),
    (
        "Rayen",
        "Mapuche",
        "Centro-sur de Chile",
        "Motivo floral inspirado en repertorios "
        "textiles mapuche; su interpretacion "
        "depende de la pieza."
    ),
    (
        "Temu",
        "Mapuche",
        "Centro-sur de Chile",
        "Interpretacion grafica de un arbol "
        "nativo; no es una reproduccion "
        "patrimonial exacta."
    ),
    (
        "Zigzag",
        "Diaguita",
        "Norte Chico de Chile",
        "Patron geometrico presente en "
        "decoraciones ceramicas; no posee "
        "un significado universal demostrado."
    ),
    (
        "Ondas",
        "Diaguita",
        "Norte Chico de Chile",
        "Lineas onduladas inspiradas en "
        "composiciones de ceramica diaguita."
    ),
    (
        "Rombos",
        "Diaguita",
        "Norte Chico de Chile",
        "Repeticion de formas geometricas "
        "inspirada en repertorios decorativos "
        "diaguitas."
    ),
    (
        "Camelidos",
        "Aymara",
        "Altiplano andino",
        "Las llamas y alpacas tienen "
        "importancia economica y cultural "
        "en comunidades andinas."
    ),
    (
        "Franjas",
        "Aymara",
        "Altiplano andino",
        "Patrones de franjas inspirados en "
        "textiles andinos; sus usos y sentidos "
        "varian segun contexto."
    ),
]

rng = np.random.default_rng(2026)

selected_motif = 0
selected_ink = 0

stamps = []
show_gallery = False
animation = None

notice = ""
notice_until = 0.0


# ==========================================
# TEXTO
# ==========================================

def txt(
    im, s, x, y,
    scale=0.55,
    color=(48, 53, 65),
    thick=1
):
    cv2.putText(
        im,
        str(s),
        (int(x), int(y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thick,
        cv2.LINE_AA
    )


def fit_text(
    im, s, x, y,
    max_w,
    scale=0.52,
    color=(48, 53, 65),
    thick=1
):
    words = s.split()
    line = ""
    lines = []

    for word in words:
        test = (line + " " + word).strip()

        width = cv2.getTextSize(
            test,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            thick
        )[0][0]

        if width > max_w and line:
            lines.append(line)
            line = word
        else:
            line = test

    if line:
        lines.append(line)

    for j, part in enumerate(lines):
        txt(
            im, part,
            x, y + 22 * j,
            scale, color, thick
        )

    return len(lines)


# ==========================================
# CREACION DE MATRICES
# ==========================================

def motif_mask(n, size=144):

    mask = np.zeros(
        (size, size),
        dtype=np.uint8
    )

    c = size // 2

    def line(a, b, thick=8):
        cv2.line(
            mask, a, b,
            255, thick, cv2.LINE_AA
        )

    if n == 0:
        # LUKUTUEL
        cv2.circle(
            mask,
            (c, c - 32),
            12, 255, -1,
            cv2.LINE_AA
        )

        line((c, c - 20), (c, c + 17))
        line((c, c - 6), (c - 32, c + 10))
        line((c, c - 6), (c + 32, c + 10))
        line((c, c + 17), (c - 30, c + 37))
        line((c, c + 17), (c + 30, c + 37))

    elif n == 1:
        # RAYEN
        cv2.circle(
            mask,
            (c, c),
            10, 255, -1,
            cv2.LINE_AA
        )

        for a in range(8):
            theta = a * math.pi / 4

            px = int(c + 33 * math.cos(theta))
            py = int(c + 33 * math.sin(theta))

            cv2.ellipse(
                mask,
                (px, py),
                (13, 18),
                a * 45,
                0, 360,
                255, 5,
                cv2.LINE_AA
            )

    elif n == 2:
        # TEMU
        line((c, c + 45), (c, c - 29))

        for level in range(3):
            yy = c - 10 - level * 15
            span = 37 - level * 7

            line(
                (c, yy),
                (c - span, yy - 20),
                6
            )

            line(
                (c, yy),
                (c + span, yy - 20),
                6
            )

    elif n == 3:
        # ZIGZAG
        for off in (-18, 18):

            pts = np.array([
                (
                    c - 48 + i * 16,
                    c + off + (-16 if i % 2 else 16)
                )
                for i in range(7)
            ], np.int32)

            cv2.polylines(
                mask, [pts],
                False, 255, 7,
                cv2.LINE_AA
            )

    elif n == 4:
        # ONDAS
        for off in (-17, 17):

            pts = np.array([
                (
                    c + dx,
                    c + off + int(
                        14 * math.sin(dx / 12)
                    )
                )
                for dx in range(-49, 50, 2)
            ], np.int32)

            cv2.polylines(
                mask, [pts],
                False, 255, 7,
                cv2.LINE_AA
            )

    elif n == 5:
        # ROMBOS
        for dx in (-26, 26):

            pts = np.array([
                (c + dx, c - 33),
                (c + dx + 25, c),
                (c + dx, c + 33),
                (c + dx - 25, c)
            ], np.int32)

            cv2.polylines(
                mask, [pts],
                True, 255, 7,
                cv2.LINE_AA
            )

    elif n == 6:
        # CAMELIDOS
        cv2.rectangle(
            mask,
            (c - 40, c - 12),
            (c + 19, c + 12),
            255, 6,
            cv2.LINE_AA
        )

        line((c + 18, c - 9), (c + 26, c - 38))
        line((c + 26, c - 38), (c + 43, c - 38))
        line((c - 29, c + 12), (c - 29, c + 46))
        line((c + 7, c + 12), (c + 7, c + 46))
        line((c - 40, c - 10), (c - 50, c - 25))

    else:
        # FRANJAS
        for dx in range(-42, 43, 16):
            line(
                (c + dx, c - 43),
                (c + dx, c + 43),
                5
            )

        for yy in (-21, 21):
            line(
                (c - 48, c + yy),
                (c + 48, c + yy),
                5
            )

    return mask


MASKS = [
    motif_mask(i)
    for i in range(8)
]


# ==========================================
# PAPEL ARTESANAL
# ==========================================

paper = np.full(
    (PH, PW, 3),
    (218, 236, 246),
    np.uint8
)

grain = rng.normal(
    0, 3.5,
    (PH, PW, 1)
)

paper = np.clip(
    paper.astype(np.float32) + grain,
    0, 255
).astype(np.uint8)

# Fibras del papel
for _ in range(1100):

    xx = int(rng.integers(0, PW))
    yy = int(rng.integers(0, PH))

    length = int(rng.integers(1, 7))

    cv2.line(
        paper,
        (xx, yy),
        (min(PW - 1, xx + length), yy),
        (203, 222, 231),
        1,
        cv2.LINE_AA
    )

canvas = paper.copy()


# ==========================================
# ESTAMPADO REALISTA
# ==========================================

def apply_stamp(n, ink_index, px, py):
    global canvas

    base = MASKS[n].astype(
        np.float32
    ) / 255.0

    size = base.shape[0]

    # Variacion de cobertura de tinta
    coarse = rng.random(
        (18, 18)
    ).astype(np.float32)

    uneven = cv2.resize(
        coarse,
        (size, size),
        interpolation=cv2.INTER_CUBIC
    )

    fine = rng.random(
        (size, size)
    ).astype(np.float32)

    alpha = base * np.clip(
        0.62
        + 0.32 * uneven
        + 0.22 * (fine - 0.5),
        0.1,
        1
    )

    # Pequenos poros sin tinta
    alpha[fine < 0.035] *= 0.12

    alpha = cv2.GaussianBlur(
        alpha,
        (3, 3),
        0.35
    )

    alpha *= 0.82

    # Desplazamiento de registro
    x1 = int(
        px - size // 2
        + rng.integers(-2, 3)
    )

    y1 = int(
        py - size // 2
        + rng.integers(-2, 3)
    )

    x2 = x1 + size
    y2 = y1 + size

    dx1 = max(0, x1)
    dy1 = max(0, y1)

    dx2 = min(PW, x2)
    dy2 = min(PH, y2)

    if dx1 >= dx2 or dy1 >= dy2:
        return

    sx1 = dx1 - x1
    sy1 = dy1 - y1

    a = alpha[
        sy1:sy1 + dy2 - dy1,
        sx1:sx1 + dx2 - dx1,
        None
    ]

    area = canvas[
        dy1:dy2,
        dx1:dx2
    ].astype(np.float32)

    pigment = np.array(
        INKS[ink_index][1],
        np.float32
    ) / 255.0

    # Superposicion multiplicativa
    area = area * (
        1 - a * (1 - pigment)
    )

    canvas[
        dy1:dy2,
        dx1:dx2
    ] = np.clip(
        area,
        0, 255
    ).astype(np.uint8)

    stamps.append(
        (n, ink_index)
    )


# ==========================================
# MESA DE TALLER
# ==========================================

def make_wood():

    wood = np.full(
        (H, W, 3),
        (125, 169, 195),
        np.uint8
    )

    noise = rng.normal(
        0, 4,
        (H, 1, 1)
    )

    wood = np.clip(
        wood.astype(np.float32) + noise,
        0, 255
    ).astype(np.uint8)

    for y in range(0, H, 82):
        cv2.line(
            wood,
            (0, y),
            (W, y),
            (112, 151, 175),
            2
        )

    return wood


WOOD = make_wood()


def card(
    im, x1, y1, x2, y2,
    fill,
    border=(114, 112, 106),
    thick=1
):

    # Sombra
    cv2.rectangle(
        im,
        (x1 + 4, y1 + 5),
        (x2 + 4, y2 + 5),
        (99, 131, 150),
        -1
    )

    cv2.rectangle(
        im,
        (x1, y1),
        (x2, y2),
        fill,
        -1
    )

    cv2.rectangle(
        im,
        (x1, y1),
        (x2, y2),
        border,
        thick
    )


def preview(
    im, n, cx, cy,
    scale=0.38,
    color=(53, 67, 88)
):

    size = max(
        20,
        int(144 * scale)
    )

    sm = cv2.resize(
        MASKS[n],
        (size, size),
        interpolation=cv2.INTER_AREA
    )

    x1 = cx - size // 2
    y1 = cy - size // 2

    x2 = x1 + size
    y2 = y1 + size

    if (
        x1 < 0 or y1 < 0
        or x2 > im.shape[1]
        or y2 > im.shape[0]
    ):
        return

    area = im[y1:y2, x1:x2]

    a = sm.astype(
        np.float32
    )[:, :, None] / 255.0

    im[y1:y2, x1:x2] = np.clip(
        area * (1 - a)
        + np.array(color) * a,
        0, 255
    ).astype(np.uint8)


# ==========================================
# PANTALLA PRINCIPAL
# ==========================================

def draw_workshop(
    cursor=None,
    pinching=False
):

    im = WOOD.copy()

    # Encabezado
    card(
        im,
        24, 17, 1176, 79,
        (218, 233, 243)
    )

    txt(
        im,
        "HUELLAS DE ORIGEN",
        47, 51,
        0.94,
        (60, 61, 105),
        2
    )

    txt(
        im,
        "TALLER DE LINOGRAFIA DIGITAL",
        685, 49,
        0.58,
        (65, 74, 87),
        2
    )

    txt(
        im,
        "1. ELIGE UNA MATRIZ",
        47, 109,
        0.60,
        (35, 49, 65),
        2
    )

    # Tarjetas culturales
    for i, (name, culture, _, _) in enumerate(MOTIFS):

        row = i // 4
        col = i % 4

        x1 = 45 + col * 282
        y1 = 126 + row * 93

        x2 = x1 + 267
        y2 = y1 + 83

        fill = (
            (196, 220, 235)
            if selected_motif == i
            else (224, 233, 238)
        )

        border = (
            (65, 74, 110)
            if selected_motif == i
            else (119, 121, 125)
        )

        card(
            im,
            x1, y1, x2, y2,
            fill,
            border,
            3 if selected_motif == i else 1
        )

        preview(
            im, i,
            x1 + 48,
            y1 + 41,
            0.37
        )

        txt(
            im,
            name.upper(),
            x1 + 91,
            y1 + 35,
            0.50,
            (46, 53, 76),
            2
        )

        txt(
            im,
            culture,
            x1 + 91,
            y1 + 61,
            0.44
        )

    # Paleta de tintas
    txt(
        im,
        "2. PREPARA LA TINTA",
        47, 339,
        0.60,
        (35, 49, 65),
        2
    )

    for i, (name, col) in enumerate(INKS):

        cx = 360 + i * 161

        cv2.circle(
            im,
            (cx, 332),
            22,
            (85, 85, 85),
            -1,
            cv2.LINE_AA
        )

        cv2.circle(
            im,
            (cx, 330),
            19,
            col,
            -1,
            cv2.LINE_AA
        )

        if i == selected_ink:
            cv2.circle(
                im,
                (cx, 330),
                28,
                (245, 247, 248),
                3,
                cv2.LINE_AA
            )

        txt(
            im,
            name,
            cx - 52,
            372,
            0.42,
            (35, 49, 65),
            2 if i == selected_ink else 1
        )

    # Sombra del papel
    cv2.rectangle(
        im,
        (
            PAPER_BOX[0] + 8,
            PAPER_BOX[1] + 9
        ),
        (
            PAPER_BOX[2] + 8,
            PAPER_BOX[3] + 9
        ),
        (93, 125, 149),
        -1
    )

    # Hoja de estampado
    im[
        PAPER_BOX[1]:PAPER_BOX[3],
        PAPER_BOX[0]:PAPER_BOX[2]
    ] = canvas

    cv2.rectangle(
        im,
        (
            PAPER_BOX[0],
            PAPER_BOX[1]
        ),
        (
            PAPER_BOX[2],
            PAPER_BOX[3]
        ),
        (126, 143, 154),
        2
    )

    txt(
        im,
        "3. ESTAMPA SOBRE EL PAPEL",
        53, 408,
        0.48,
        (120, 125, 135)
    )

    # Ficha cultural
    card(
        im,
        45, 714, 1155, 831,
        (226, 237, 241)
    )

    name, culture, origin, meaning = MOTIFS[
        selected_motif
    ]

    txt(
        im,
        name.upper()
        + "  |  "
        + culture.upper()
        + "  |  "
        + origin,
        63, 748,
        0.61,
        (52, 59, 84),
        2
    )

    fit_text(
        im,
        meaning,
        63, 777,
        1050,
        0.49
    )

    txt(
        im,
        "Interpretaciones graficas educativas; "
        "no son matrices historicas exactas.",
        63, 815,
        0.39,
        (94, 99, 108)
    )

    txt(
        im,
        "Pellizca: seleccionar/estampar   |   "
        "C: limpiar   |   S: guardar   |   "
        "E: exposicion   |   Q: salir",
        65, 869,
        0.47,
        (244, 244, 244),
        2
    )

    # ======================================
    # ANIMACION DE PRESION
    # ======================================

    if animation is not None:

        n, ink, px, py, t0 = animation

        elapsed = time.monotonic() - t0

        if elapsed < 0.42:

            phase = elapsed / 0.42

            scale = (
                0.74
                + 0.26 * math.sin(
                    math.pi * min(phase, 1)
                )
            )

            radius = int(
                65 * scale
            )

            sx = PAPER_BOX[0] + px
            sy = PAPER_BOX[1] + py

            # Sombra de matriz
            cv2.circle(
                im,
                (sx + 5, sy + 7),
                radius,
                (144, 151, 154),
                -1,
                cv2.LINE_AA
            )

            # Matriz
            cv2.circle(
                im,
                (sx, sy),
                radius,
                (65, 85, 112),
                -1,
                cv2.LINE_AA
            )

            cv2.circle(
                im,
                (sx, sy),
                radius - 7,
                INKS[ink][1],
                4,
                cv2.LINE_AA
            )

            preview(
                im,
                n,
                sx, sy,
                0.60,
                (235, 238, 242)
            )

            txt(
                im,
                "PRESIONANDO...",
                max(55, sx - 84),
                max(410, sy - 80),
                0.49,
                (40, 55, 75),
                2
            )

    # Mensajes de confirmacion
    if (
        notice
        and time.monotonic() < notice_until
    ):
        cv2.rectangle(
            im,
            (370, 676),
            (830, 708),
            (44, 102, 74),
            -1
        )

        txt(
            im,
            notice,
            385, 698,
            0.54,
            (255, 255, 255),
            2
        )

    # Cursor de mano
    if cursor:

        x, y = cursor

        color = (
            (54, 171, 72)
            if pinching
            else (50, 120, 230)
        )

        cv2.circle(
            im,
            (x, y),
            15,
            color,
            3,
            cv2.LINE_AA
        )

        cv2.circle(
            im,
            (x, y),
            4,
            (255, 255, 255),
            -1,
            cv2.LINE_AA
        )

    return im


# ==========================================
# EXPOSICION FINAL
# ==========================================

def draw_gallery():

    im = WOOD.copy()

    card(
        im,
        35, 25, 1165, 870,
        (221, 232, 240)
    )

    txt(
        im,
        "EXPOSICION FINAL",
        370, 86,
        1.05,
        (53, 61, 95),
        2
    )

    txt(
        im,
        "HUELLAS DE ORIGEN  /  OBRA DIGITAL",
        329, 123,
        0.60
    )

    # Marco
    cv2.rectangle(
        im,
        (92, 164),
        (1108, 708),
        (71, 101, 130),
        -1
    )

    # Paspartu
    cv2.rectangle(
        im,
        (111, 183),
        (1089, 689),
        (225, 230, 231),
        -1
    )

    # Obra final
    view = cv2.resize(
        canvas,
        (940, 400),
        interpolation=cv2.INTER_AREA
    )

    im[236:636, 130:1070] = view

    cv2.rectangle(
        im,
        (130, 236),
        (1070, 636),
        (125, 129, 131),
        1
    )

    txt(
        im,
        "Obra creada mediante estampado digital "
        "de matrices inspiradas en motivos culturales.",
        113, 747,
        0.48
    )

    txt(
        im,
        "Impresiones: "
        + str(len(stamps))
        + "   |   Colores usados: "
        + str(
            len(set(
                i for _, i in stamps
            ))
        ),
        113, 779,
        0.51,
        (51, 65, 91),
        2
    )

    txt(
        im,
        "S: guardar PNG   |   E: volver al taller   |   "
        "C: limpiar   |   Q: salir",
        113, 831,
        0.49,
        (56, 68, 91),
        2
    )

    if (
        notice
        and time.monotonic() < notice_until
    ):
        txt(
            im,
            notice,
            685, 779,
            0.51,
            (46, 118, 60),
            2
        )

    return im


# ==========================================
# GUARDAR IMAGEN PNG
# ==========================================

def save_png():

    global notice
    global notice_until

    filename = (
        "huellas_de_origen_"
        + datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        + ".png"
    )

    ok = cv2.imwrite(
        filename,
        canvas
    )

    if ok:
        notice = "GUARDADO: " + filename

        print(
            "Imagen guardada en:",
            os.path.abspath(filename)
        )

    else:
        notice = "ERROR AL GUARDAR PNG"

    notice_until = (
        time.monotonic() + 3.0
    )


# ==========================================
# ABRIR CAMARA EN MAC
# ==========================================

def open_camera():

    # Primero intenta camara 1,
    # luego camara 0 y finalmente 2.

    for index in (1, 0, 2):

        cap = cv2.VideoCapture(
            index,
            cv2.CAP_AVFOUNDATION
        )

        if cap.isOpened():

            ok, _ = cap.read()

            if ok:
                print(
                    "Camara seleccionada:",
                    index
                )
                return cap

        cap.release()

    raise RuntimeError(
        "No se pudo abrir una camara. "
        "Revisa los permisos de macOS."
    )


# ==========================================
# MEDIAPIPE Y CONTROL POR GESTOS
# ==========================================

def main():

    global selected_motif
    global selected_ink
    global canvas
    global animation
    global show_gallery
    global notice
    global notice_until

    if not os.path.isfile(
        "hand_landmarker.task"
    ):
        raise FileNotFoundError(
            "Falta hand_landmarker.task "
            "en esta carpeta."
        )

    base = mp.tasks.BaseOptions

    opts = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base(
            model_asset_path="hand_landmarker.task"
        ),
        running_mode=(
            mp.tasks.vision.RunningMode.VIDEO
        ),
        num_hands=1
    )

    cap = open_camera()

    last_pinch = False
    last_action = 0.0

    start = time.monotonic()
    last_timestamp = -1

    window = (
        "Huellas de Origen - "
        "Taller de Linografia"
    )

    cv2.namedWindow(
        window,
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        window,
        W, H
    )

    try:

        with mp.tasks.vision.HandLandmarker.create_from_options(
            opts
        ) as detector:

            while True:

                ok, frame = cap.read()

                if not ok:
                    print(
                        "No se pudo leer la camara."
                    )
                    break

                frame = cv2.flip(
                    frame, 1
                )

                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                data = mp.Image(
                    image_format=(
                        mp.ImageFormat.SRGB
                    ),
                    data=rgb
                )

                timestamp = max(
                    last_timestamp + 1,
                    int(
                        (
                            time.monotonic()
                            - start
                        ) * 1000
                    )
                )

                last_timestamp = timestamp

                result = detector.detect_for_video(
                    data,
                    timestamp
                )

                now = time.monotonic()

                # Terminar animacion
                if (
                    animation is not None
                    and now - animation[4] >= 0.42
                ):

                    n, ink, px, py, _ = animation

                    apply_stamp(
                        n, ink, px, py
                    )

                    animation = None

                cursor = None
                pinching = False

                if result.hand_landmarks:

                    hand = result.hand_landmarks[0]

                    x = max(
                        0,
                        min(
                            W - 1,
                            int(hand[8].x * W)
                        )
                    )

                    y = max(
                        0,
                        min(
                            H - 1,
                            int(hand[8].y * H)
                        )
                    )

                    cursor = (x, y)

                    dist = math.hypot(
                        hand[8].x - hand[4].x,
                        hand[8].y - hand[4].y
                    )

                    pinching = dist < 0.055

                    new_gesture = (
                        not show_gallery
                        and pinching
                        and not last_pinch
                        and now - last_action > 0.4
                    )

                    if new_gesture:

                        # Seleccionar matriz
                        if 126 <= y < 302:

                            for i in range(8):

                                row, col = divmod(
                                    i, 4
                                )

                                xx = 45 + col * 282
                                yy = 126 + row * 93

                                if (
                                    xx <= x <= xx + 267
                                    and yy <= y <= yy + 83
                                ):
                                    selected_motif = i
                                    break

                        # Seleccionar tinta
                        elif 302 <= y <= 373:

                            for i in range(
                                len(INKS)
                            ):

                                cx = 360 + i * 161

                                if (
                                    abs(x - cx) <= 31
                                    and abs(y - 330) <= 35
                                ):
                                    selected_ink = i
                                    break

                        # Estampar sobre el papel
                        elif (
                            PAPER_BOX[0] + 5
                            <= x
                            < PAPER_BOX[2] - 5
                            and PAPER_BOX[1] + 10
                            <= y
                            < PAPER_BOX[3] - 5
                            and animation is None
                        ):

                            animation = (
                                selected_motif,
                                selected_ink,
                                x - PAPER_BOX[0],
                                y - PAPER_BOX[1],
                                now
                            )

                        last_action = now

                    last_pinch = pinching

                else:
                    last_pinch = False

                # Elegir pantalla
                if show_gallery:
                    screen = draw_gallery()
                else:
                    screen = draw_workshop(
                        cursor,
                        pinching
                    )

                cv2.imshow(
                    window,
                    screen
                )

                key = cv2.waitKey(1) & 0xFF

                # SALIR
                if key == ord("q"):
                    break

                # BORRAR
                elif key == ord("c"):

                    canvas = paper.copy()
                    stamps.clear()
                    animation = None

                    notice = "PAPEL LIMPIO"
                    notice_until = (
                        time.monotonic() + 2
                    )

                # GUARDAR
                elif key == ord("s"):
                    save_png()

                # MODO EXPOSICION
                elif key == ord("e"):
                    show_gallery = (
                        not show_gallery
                    )

                # Cerrar con la X
                if cv2.getWindowProperty(
                    window,
                    cv2.WND_PROP_VISIBLE
                ) < 1:
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()


# ==========================================
# INICIAR PROGRAMA
# ==========================================

if __name__ == "__main__":
    main()
