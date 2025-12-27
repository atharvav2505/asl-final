from flask import Flask, render_template, Response, request, jsonify
import cv2
import alphabets
import dynamic
import atexit

# 🔎 Confirm which alphabets.py is being used
print("USING alphabets.py FROM:", alphabets.__file__)

app = Flask(__name__)

# Camera
camera = cv2.VideoCapture(0)

# Modes: "alphabet" | "dynamic"
mode = "alphabet"

# Load alphabet model once
model = alphabets.load_model()

# Sentence state (used only in dynamic mode)
sentence = []
last_word = None


def generate_frames():
    global mode, sentence, last_word

    while True:
        success, frame = camera.read()
        if not success:
            break

        if mode == "alphabet":
            # 🔤 Alphabet recognition → DISPLAY ONLY
            # All drawing is handled inside alphabets.py
            _, _, frame = alphabets.predict_alphabet(frame, model)

        elif mode == "dynamic":
            # ✋ Custom gesture recognition
            frame, label = dynamic.predict_dynamic(frame)

            # Add word only when gesture changes
            if label and label != last_word:
                sentence.append(label)
                last_word = label

        # Encode frame for streaming
        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/set_mode/<selected>")
def set_mode(selected):
    global mode, last_word
    mode = selected
    last_word = None
    print(f"[DEBUG] MODE SWITCHED TO: {mode}")
    return ("", 204)


@app.route("/save_gesture", methods=["POST"])
def save_gesture():
    name = request.form.get("gesture_name")
    success, msg = dynamic.save_gesture(name)
    return msg, 200 if success else 400


@app.route("/get_sentence")
def get_sentence():
    return jsonify({"sentence": sentence})


@app.route("/delete_word/<int:index>", methods=["DELETE"])
def delete_word(index):
    try:
        sentence.pop(index)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/clear_sentence", methods=["POST"])
def clear_sentence():
    sentence.clear()
    return jsonify({"success": True})


# 🔒 Proper camera cleanup
@atexit.register
def cleanup():
    try:
        camera.release()
        print("[DEBUG] Camera released")
    except:
        pass


if __name__ == "__main__":
    app.run(debug=True)
