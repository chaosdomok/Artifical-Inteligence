import cv2
import numpy as np
import math

# Funkcja do rozpoznawania palców
def detect_fingers(contour, hull):
    # Oblicz odległości między konturem a wypukłością
    defects = cv2.convexityDefects(contour, hull)
    if defects is None:
        return 0
    
    finger_count = 0

    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])

        # Obliczamy długości boków trójkąta
        a = math.dist(start, end)
        b = math.dist(start, far)
        c = math.dist(end, far)

        # Prawo cosinusów do znalezienia kąta
        angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))

        # Jeżeli kąt jest mniejszy niż 90 stopni, uznajemy to za uniesiony palec
        if angle <= math.pi / 2:
            finger_count += 1

    # Zwraca liczbę wykrytych palców
    return finger_count + 1

# Uruchamiamy kamerę
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Odbijamy obraz poziomo
    frame = cv2.flip(frame, 1)

    # Definiujemy obszar zainteresowania (ROI)
    roi = frame[100:400, 100:400]
    
    # Zmieniamy obraz na szary
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Zmieniamy na obraz binarny przez progowanie
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Znajdujemy kontury
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Wybieramy największy kontur (który jest ręką)
        contour = max(contours, key=lambda x: cv2.contourArea(x))
        
        # Znajdujemy wypukłość konturu
        hull = cv2.convexHull(contour, returnPoints=False)

        if hull is not None:
            finger_count = detect_fingers(contour, hull)
            print(f"Liczba palców: {finger_count}")

            # Rysujemy kontur ręki i wypukłość
            cv2.drawContours(roi, [contour], -1, (0, 255, 0), 2)
    
    # Wyświetlamy okno z obrazem
    cv2.imshow("Frame", frame)
    cv2.imshow("Threshold", thresh)

    # Zakończenie po wciśnięciu klawisza 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
