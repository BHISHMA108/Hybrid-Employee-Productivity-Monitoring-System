import sys
import logging
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import time
import threading
import cv2
import numpy as np
from flask import Flask, jsonify
from flask_cors import CORS
from keras.models import load_model
from PIL import Image, ImageOps
from pynput import keyboard, mouse
import pygetwindow as gw

# --- CONFIGURATION ---
MODEL_PATH = r"D:\Final Year\Sem 2\Major Project\Model Project\Model B (Keras Tensorflow)\keras_model_with_5k_images.h5"
IDLE_THRESHOLD = 2
BREAK_THRESHOLD = 3

logging.basicConfig(level=logging.INFO, format='%(message)s')
app = Flask(__name__)
CORS(app)

# Load Model
try:
    model = load_model(MODEL_PATH)
    logging.info("[OK] Hybrid H5 Model loaded successfully for testing.")
except Exception as e:
    logging.error(f"[ERR] Failed to load model: {e}")
    exit(1)

# Global Metrics
monitoring = False
active_duration = 0
inactive_duration = 0
total_break_time = 0
break_counter = 0
tab_switch_count = -1
previous_window = None
last_activity_time = time.time()
session_start_time = 0
in_break = False
break_start_time = None
break_counted = False
current_accuracy = 0.0

def convert_to_hms(seconds):
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{secs:02}"

def update_last_activity():
    global last_activity_time
    last_activity_time = time.time()

def draw_transparent_hud(frame, metrics, status_color, accuracy_pct):
    """
    Draw a fully transparent HUD overlay:
    - A semi-transparent dark panel behind text only (alpha blending)
    - Text rendered on top so the face/background is always visible
    """
    # Panel dimensions
    pad_x, pad_y = 15, 12
    line_h = 32
    panel_w = 340
    panel_h = pad_y * 2 + len(metrics) * line_h

    # Create a dark overlay copy only for the panel region
    panel = frame.copy()
    x1, y1, x2, y2 = 10, 10, 10 + panel_w, 10 + panel_h
    cv2.rectangle(panel, (x1, y1), (x2, y2), (20, 20, 20), -1)

    # Blend panel — alpha=0.45 keeps face visible through HUD
    alpha = 0.45
    cv2.addWeighted(panel, alpha, frame, 1 - alpha, 0, frame)

    # Draw a thin border around the panel
    cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 1)

    # Render each metric as a plain text line
    for i, (label, value, highlight) in enumerate(metrics):
        tx = x1 + pad_x
        ty = y1 + pad_y + i * line_h + 20

        if i == 0:
            # Title row — cyan accent, slightly larger
            cv2.putText(frame, label, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 220), 2, cv2.LINE_AA)
        else:
            txt_color = status_color if highlight else (210, 210, 210)
            cv2.putText(frame, f"{label}: {value}", (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, txt_color, 1, cv2.LINE_AA)

def monitor_activity():
    global monitoring, active_duration, inactive_duration, total_break_time, \
           break_counter, tab_switch_count, previous_window, last_activity_time, \
           session_start_time, in_break, break_start_time, break_counted, current_accuracy

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("[ERR] Cannot open webcam.")
        monitoring = False
        return

    k_listener = keyboard.Listener(on_press=lambda _: update_last_activity())
    m_listener = mouse.Listener(on_move=lambda x, y: update_last_activity())
    k_listener.start()
    m_listener.start()

    monitoring = True
    session_start_time = time.time()
    last_check_time = session_start_time

    logging.info("[*] Live Monitoring Window is opening...")

    while monitoring:
        current_time = time.time()
        delta_time = current_time - last_check_time

        ret, frame = cap.read()
        if not ret:
            continue

        # --- 1. Prediction & Confidence ---
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        pil_img = ImageOps.fit(pil_img, (224, 224), Image.Resampling.LANCZOS)
        input_data = np.expand_dims(np.asarray(pil_img, dtype=np.float32) / 255.0, axis=0)

        predictions = model.predict(input_data, verbose=0)
        prediction_idx = int(np.argmax(predictions))
        face_present = (prediction_idx == 0)
        face_conf     = float(predictions[0][0]) * 100   # confidence for Face
        non_face_conf = float(predictions[0][1]) * 100   # confidence for Non-Face
        current_accuracy = face_conf if face_present else non_face_conf

        # --- 2. Status Logic ---
        time_since_input = current_time - last_activity_time

        if face_present:
            in_break = False
            break_counted = False
            if time_since_input > IDLE_THRESHOLD:
                inactive_duration += delta_time
                status = "IDLE"
                status_color = (0, 220, 220)   # Cyan
            else:
                active_duration += delta_time
                status = "ACTIVE"
                status_color = (0, 220, 80)    # Green
        else:
            total_break_time += delta_time
            status = "AWAY"
            status_color = (60, 80, 220)       # Red-ish
            if not in_break:
                in_break = True
                break_start_time = current_time
            elif (current_time - break_start_time) >= BREAK_THRESHOLD and not break_counted:
                break_counter += 1
                break_counted = True

        # --- 3. Tab Switch Detection ---
        try:
            aw = gw.getActiveWindow()
            if aw and aw.title != previous_window:
                tab_switch_count += 1
                previous_window = aw.title
        except Exception:
            pass

        # --- 4. Build HUD metrics list ---
        # Each entry: (label, value, highlight_with_status_color)
        session_s = current_time - session_start_time
        productivity = (active_duration / session_s * 100) if session_s > 0 else 0

        metrics = [
            ("Model Live Monitoring Status", "",                    False),  # Title row
            ("Face Confidence",              f"{face_conf:.1f}%",     False),
            ("Non-Face Confidence",          f"{non_face_conf:.1f}%", False),
            ("Status",                       status,                 True),
            ("Session",                      convert_to_hms(session_s),          False),
            ("Active",                       convert_to_hms(active_duration),    False),
            ("Idle",                         convert_to_hms(inactive_duration),  False),
            ("Break Time",                   convert_to_hms(total_break_time),   False),
            ("Break Count",                  str(break_counter),                 False),
            ("Tab Switches",                 str(max(0, tab_switch_count)),       False),
            ("Productivity",                 f"{productivity:.1f}%",             False),
        ]

        # --- 5. Draw transparent HUD ---
        draw_transparent_hud(frame, metrics, status_color, current_accuracy)

        cv2.imshow("EfficienSee - Live HUD", frame)
        last_check_time = current_time

        # ESC to quit
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    k_listener.stop()
    m_listener.stop()
    monitoring = False


@app.route("/start")
def start():
    if not monitoring:
        threading.Thread(target=monitor_activity, daemon=True).start()
        return jsonify({"status": "HUD Started", "accuracy_tracking": "Enabled"})
    return jsonify({"status": "Already Running"})


@app.route("/status")
def status():
    return jsonify({
        "monitoring": monitoring,
        "accuracy": round(current_accuracy, 2),
        "active": round(active_duration, 1),
        "inactive": round(inactive_duration, 1),
        "break_time": round(total_break_time, 1),
        "breaks": break_counter,
        "tab_switches": max(0, tab_switch_count),
    })


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  EfficienSee - Live Monitoring HUD")
    print("=" * 50)
    print("  [OK] Model loaded. Starting monitoring...")
    print("  [*]  Webcam will open automatically.")
    print("  [*]  API running at http://127.0.0.1:5001")
    print("  [*]  Press ESC inside the HUD to stop.")
    print("=" * 50 + "\n")

    # Auto-start monitoring immediately when server launches
    threading.Thread(target=monitor_activity, daemon=True).start()

    app.run(host="0.0.0.0", port=5001, debug=False)