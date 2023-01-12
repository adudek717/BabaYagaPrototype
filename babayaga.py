"""
Projekt:
    Prototyp maszyny do gry w "Baba Jaga patrzy" posiadający funkcjonalności:
    - Rysowanie celownika na twarzy celu
    - Celowanie tylko w ruchome cele

Przygotowanie środowiska:
    Język Python oraz biblioteka opencv-python(pip install opencv-python).

Działanie aplikacji:
    W folderze z aplikacją nalezy umieścić plik wideo w formacie mp4 oraz nazwać go "video.mp4".
    Następnie wystarczy uruchomić naszą aplikację(python babayaga.py).

Autorzy:
    Aleksander Dudek s20155
    Jakub Słomiński  s18552
"""

import cv2

# Load the video
video_file = "video.mp4"
capture = cv2.VideoCapture(video_file)

# Initialize face detection classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Set up previous frame for face movement check
_, prev_frame = capture.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while True:
    # Read current frame
    success, frame = capture.read()

    # Check if video has reached the end
    if not success:
        break

    # Convert current frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in current frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Check if face has moved by comparing the previous frame
        face_movement = cv2.absdiff(
            prev_gray[y:y+h, x:x+w], gray[y:y+h, x:x+w])
        if cv2.countNonZero(face_movement) > 17000:
            # Draw crosshair on face
            cv2.line(frame, (x + w//2, y), (x + w//2, y + h), (0, 0, 255), 2)
            cv2.line(frame, (x, y + h//2), (x + w, y + h//2), (0, 0, 255), 2)

    # Show frame
    cv2.imshow("Frame", frame)

    # Update previous frame
    prev_gray = gray

    # Wait for user to press a key
    key = cv2.waitKey(1)
    if key == 27:  # Esc key
        break

# Release capture and close the window
capture.release()
cv2.destroyAllWindows()
