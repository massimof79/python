import cv2
import face_recognition
import numpy as np
import pickle
import os
from sklearn.svm import SVC

MODEL_FILE = "face_model.pkl"
DATA_FILE = "face_data.pkl"

def acquire_faces_from_webcam(name, num_samples=40):
    video = cv2.VideoCapture(0)
    encodings = []

    print(f"Inizio acquisizione per {name}. Premi Q per interrompere.")

    while len(encodings) < num_samples:
        ret, frame = video.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb_frame)

        if len(locations) == 1:
            encoding = face_recognition.face_encodings(rgb_frame, locations)[0]
            encodings.append(encoding)
            cv2.putText(frame, f"Acquisite: {len(encodings)}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Acquisizione volto", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
    return encodings

def train_from_webcam():
    name = input("Inserisci il nome della persona: ").strip()

    if not name:
        print("Nome non valido.")
        return

    encodings = acquire_faces_from_webcam(name)

    if len(encodings) == 0:
        print("Nessun volto acquisito.")
        return

    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "rb") as f:
            data = pickle.load(f)
    else:
        data = {"X": [], "y": []}

    for enc in encodings:
        data["X"].append(enc)
        data["y"].append(name)

    with open(DATA_FILE, "wb") as f:
        pickle.dump(data, f)

    model = SVC(kernel="linear", probability=True)
    model.fit(data["X"], data["y"])

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

    print("Addestramento completato e modello salvato.")

def recognize_person():
    if not os.path.exists(MODEL_FILE):
        print("Modello non trovato. Esegui prima l’addestramento.")
        return

    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)

    video = cv2.VideoCapture(0)
    print("Riconoscimento attivo. Premi Q per uscire.")

    while True:
        ret, frame = video.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb_frame)
        encodings = face_recognition.face_encodings(rgb_frame, locations)

        for (top, right, bottom, left), encoding in zip(locations, encodings):
            name = model.predict([encoding])[0]
            confidence = max(model.predict_proba([encoding])[0])

            label = f"{name} ({confidence:.2f})"
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, label, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Riconoscimento facciale", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

def menu():
    while True:
        print("\nMENU")
        print("1) Addestramento tramite webcam")
        print("2) Riconoscimento persona")
        print("0) Esci")

        choice = input("Scelta: ")

        if choice == "1":
            train_from_webcam()
        elif choice == "2":
            recognize_person()
        elif choice == "0":
            break
        else:
            print("Scelta non valida.")

if __name__ == "__main__":
    menu()
