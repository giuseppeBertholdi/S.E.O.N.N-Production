import cv2
import onnxruntime as ort
import numpy as np

# Carrega o modelo
session = ort.InferenceSession("emotion-ferplus-8.onnx")

emotion_table = ['neutral','happiness','surprise','sadness','anger','disgust','fear','contempt']

def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(gray, (64, 64)).astype(np.float32)
    face = np.expand_dims(np.expand_dims(face, 0), 0)
    return face

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    input_data = preprocess(frame)
    outputs = session.run(None, {"Input3": input_data})
    probs = softmax(outputs[0][0])
    emotion = emotion_table[np.argmax(probs)]

    cv2.putText(frame, f"Emotion: {emotion}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("FER+ Emotion Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
