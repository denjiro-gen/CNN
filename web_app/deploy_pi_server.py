#!/usr/bin/env python3
"""
Deployable Chamber Web Server & Raspberry Pi Camera Bridge
Serves the web UI and live video stream directly from Raspberry Pi Camera (Picamera2 / V4L2 / OpenCV).
"""
import os
import io
import time
import json
import base64
import threading
import datetime
import cv2
import numpy as np
from PIL import Image
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
import mimetypes

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Try loading TFLite / LiteRT
Interpreter = None
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        try:
            from tensorflow.lite.python.interpreter import Interpreter
        except ImportError:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter

MODEL_PATH = os.path.join(CURRENT_DIR, "chamber_model.tflite")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "chamber_model.tflite"

interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
INPUT_SHAPE = input_details[0]['shape']

# Camera Stream State
class CameraStream:
    def __init__(self):
        self.cap = None
        self.is_running = False
        self.latest_frame = None
        self.latest_prediction = {
            "label": "Ready",
            "confidence": 0.0,
            "is_aegypti": False,
            "p_aegypti": 0.0,
            "p_other": 1.0,
            "box": [0.05, 0.05, 0.95, 0.95]
        }
        self.lock = threading.Lock()

    def start(self, camera_id=0):
        if self.is_running:
            return
        # Try Raspberry Pi Picamera2 / V4L2 / USB
        self.cap = cv2.VideoCapture(camera_id)
        # Optimizations for Raspberry Pi performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 15)
        self.is_running = True
        threading.Thread(target=self._update_loop, daemon=True).start()

    def _update_loop(self):
        last_infer = 0
        while self.is_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            
            with self.lock:
                self.latest_frame = frame.copy()

            # Run inference ~4 times/sec to conserve Pi CPU
            now = time.time()
            if now - last_infer > 0.25:
                last_infer = now
                self._run_inference(frame)

    def _run_inference(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        expected_dtype = input_details[0]['dtype'] if input_details else np.float32

        # Dual-scale input (full frame + center region)
        resized_full = cv2.resize(rgb, (128, 128))
        if expected_dtype == np.uint8:
            arr_full = np.expand_dims(resized_full.astype(np.uint8), 0)
        else:
            arr_full = np.expand_dims(resized_full.astype(np.float32) / 255.0, 0)
        interpreter.set_tensor(input_details[0]['index'], arr_full)
        interpreter.invoke()
        probs_full = interpreter.get_tensor(output_details[0]['index'])[0]

        # Centered crop
        crop = rgb[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)]
        resized_crop = cv2.resize(crop, (128, 128))
        if expected_dtype == np.uint8:
            arr_crop = np.expand_dims(resized_crop.astype(np.uint8), 0)
        else:
            arr_crop = np.expand_dims(resized_crop.astype(np.float32) / 255.0, 0)
        interpreter.set_tensor(input_details[0]['index'], arr_crop)
        interpreter.invoke()
        probs_crop = interpreter.get_tensor(output_details[0]['index'])[0]

        p_ae = float((probs_crop[0] * 0.6) + (probs_full[0] * 0.4))
        p_ot = float(1.0 - p_ae)
        is_ae = (p_ae >= 0.40)

        with self.lock:
            self.latest_prediction = {
                "label": "Aedes aegypti" if is_ae else "Other",
                "confidence": round(float(p_ae if is_ae else p_ot) * 100, 1),
                "is_aegypti": is_ae,
                "p_aegypti": round(p_ae, 4),
                "p_other": round(p_ot, 4),
                "box": [0.1, 0.1, 0.9, 0.9]
            }

    def get_jpeg(self):
        with self.lock:
            if self.latest_frame is None:
                return None
            ret, jpeg = cv2.imencode('.jpg', self.latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            return jpeg.tobytes() if ret else None

    def stop(self):
        self.is_running = False
        if self.cap:
            self.cap.release()

camera_stream = CameraStream()

class PiDeployHandler(BaseHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ('/', '/index.html'):
            idx_file = os.path.join(CURRENT_DIR, "index.html")
            with open(idx_file, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        elif path == '/api/pi/stream':
            # MJPEG stream from Raspberry Pi Camera
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            try:
                while True:
                    frame = camera_stream.get_jpeg()
                    if frame:
                        self.wfile.write(b'--frame\r\n')
                        self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    time.sleep(0.06)
            except Exception:
                pass

        elif path == '/api/pi/latest':
            with camera_stream.lock:
                pred = camera_stream.latest_prediction
            body = json.dumps(pred).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        else:
            safe = os.path.normpath(os.path.join(CURRENT_DIR, path.lstrip('/')))
            if safe.startswith(CURRENT_DIR) and os.path.isfile(safe):
                mime, _ = mimetypes.guess_type(safe)
                with open(safe, 'rb') as f:
                    data = f.read()
                self.send_response(200)
                self.send_header('Content-Type', mime or 'application/octet-stream')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_error(404)

def main():
    print("=" * 65)
    print("  Raspberry Pi Chamber Mosquito Classifier Web App")
    print("  Deployable Server with direct Pi Camera Hardware Hook")
    print("  Port: 5000 (accessible across LAN)")
    print("=" * 65)
    
    # Auto-start Pi camera hardware detection
    camera_stream.start(0)

    server = HTTPServer(('0.0.0.0', 5000), PiDeployHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        camera_stream.stop()
        server.server_close()

if __name__ == '__main__':
    main()
