import os
import pickle
import numpy as np
from flask import Flask, render_template, request, send_file
from music21 import stream, note

# ========================
# BASIC SETTINGS
# ========================

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
NOTE_DATA = "note_data.pkl"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ========================
# LOAD NOTE DATA SAFELY
# ========================

if os.path.exists(NOTE_DATA):
    with open(NOTE_DATA, "rb") as f:
        notes = pickle.load(f)
else:
    # fallback notes if file missing
    notes = ["C4", "E4", "G4", "A4", "B4"]

# FIX for dict/type errors
clean_notes = []
for n in notes:
    if isinstance(n, str):
        clean_notes.append(n)

pitchnames = sorted(list(set(clean_notes)))

if len(pitchnames) == 0:
    pitchnames = ["C4", "D4", "E4", "F4", "G4"]

# ========================
# MUSIC GENERATION (SAFE MODE)
# ========================

def generate_notes(length=100):
    """Simple random generation (stable mode)"""
    return np.random.choice(pitchnames, length)

def save_midi(generated_notes, filepath):
    output_notes = []

    offset = 0
    for pattern in generated_notes:
        new_note = note.Note(pattern)
        new_note.offset = offset
        output_notes.append(new_note)
        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write("midi", fp=filepath)

# ========================
# ROUTES
# ========================

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():

    generated_notes = generate_notes()

    output_path = os.path.abspath(
        os.path.join(OUTPUT_FOLDER, "generated.mid")
    )

    save_midi(generated_notes, output_path)

    return send_file(
        output_path,
        as_attachment=True,
        download_name="generated.mid",
        mimetype="audio/midi"
    )

# ========================
# RUN
# ========================

if __name__ == "__main__":
    app.run(debug=True)