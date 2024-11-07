import cv2
import mediapipe as mp
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Инициализация модулей для обнаружения рук и управления громкостью
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Доступ к системному контролю громкости через Pycaw
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volume_range = volume.GetVolumeRange()
min_vol = volume_range[0]
max_vol = volume_range[1]

# Запуск видеопотока
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Преобразование кадра в RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # Проверка, обнаружены ли руки
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Координаты большого пальца (точка 4) и указательного пальца (точка 8)
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            
            # Преобразование координат в пиксели
            h, w, _ = frame.shape
            thumb_pos = (int(thumb_tip.x * w), int(thumb_tip.y * h))
            index_pos = (int(index_tip.x * w), int(index_tip.y * h))

            # Отображение кружков на пальцах
            cv2.circle(frame, thumb_pos, 10, (0, 255, 0), cv2.FILLED)
            cv2.circle(frame, index_pos, 10, (0, 255, 0), cv2.FILLED)
            cv2.line(frame, thumb_pos, index_pos, (255, 0, 0), 3)

            # Расчет расстояния между пальцами
            length = np.hypot(index_pos[0] - thumb_pos[0], index_pos[1] - thumb_pos[1])

            # Переводим расстояние в диапазон громкости
            vol = np.interp(length, [30, 300], [min_vol, max_vol])
            volume.SetMasterVolumeLevel(vol, None)

            # Отображение уровня громкости на экране
            vol_bar = np.interp(length, [30, 300], [400, 150])
            cv2.rectangle(frame, (50, 150), (85, 400), (0, 255, 0), 2)
            cv2.rectangle(frame, (50, int(vol_bar)), (85, 400), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, f'Vol: {int(vol)}', (40, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Отображение кадра
    cv2.imshow("Volume Control", frame)

    # Выход при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
