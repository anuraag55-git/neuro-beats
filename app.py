import os
import numpy as np
import pickle
import tensorflow as tf
from flask import Flask, render_template, request, send_file
from music21 import note, stream

# =========================
# CONFIG
# =========================
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
SEQUENCE_LENGTH = 50

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# =========================
# LOAD NOTE DATA
# =========================
with open("note_data.pkl", "rb") as f:
    note_to_int, int_to_note, unique_notes = pickle.load(f)

n_vocab = len(unique_notes)

# =========================
# BUILD MODEL (MATCH TRAINING)
# =========================
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(SEQUENCE_LENGTH, 1)),
    tf.keras.layers.LSTM(256, return_sequences=True),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.LSTM(256),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(n_vocab, activation="softmax")
])

model.compile(loss="categorical_crossentropy", optimizer="adam")

# =========================
# LOAD WEIGHTS (SAFE)
# =========================
model.load_weights("music.weights.h5", by_name=True, skip_mismatch=True)
print("✅ Model loaded")

# =========================
# MUSIC GENERATION (FIXED)
# =========================
def generate_music(length=300, temperature=0.9):
    # random seed (NO crash possible)
    pattern = np.random.randint(0, n_vocab, size=(SEQUENCE_LENGTH,)).tolist()
    output_notes = []

    for _ in range(length):
        x = np.reshape(pattern, (1, SEQUENCE_LENGTH, 1)) / float(n_vocab)
        prediction = model.predict(x, verbose=0)[0]

        # temperature sampling (prevents same-note loop)
        preds = np.log(prediction + 1e-9) / temperature
        exp_preds = np.exp(preds)
        probs = exp_preds / np.sum(exp_preds)

        index = np.random.choice(len(probs), p=probs)
        result = int_to_note[index]

        output_notes.append(result)
        pattern.append(index)
        pattern = pattern[1:]

    return output_notes

# =========================
# MIDI SAVE
# =========================
def save_midi(notes_list, filename):
    midi_notes = []
    for n in notes_list:
        try:
            midi_notes.append(note.Note(n))
        except:
            continue

    midi_stream = stream.Stream(midi_notes)
    midi_stream.write("midi", fp=filename)

# =========================
# FLASK APP
# =========================
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    temperature = float(request.form.get("temperature", 0.9))

    generated_notes = generate_music(
        length=300,
        temperature=temperature
    )

    output_path = os.path.join(OUTPUT_FOLDER, "generated.mid")
    save_midi(generated_notes, output_path)

    return send_file(output_path, as_attachment=True)

# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(debug=True)
