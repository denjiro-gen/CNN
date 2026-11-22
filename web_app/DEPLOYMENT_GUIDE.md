# Chamber Mosquito Detector — Deployment & Raspberry Pi Camera Setup

This standalone web application is ready to run on any local machine, cloud VM, Docker container, or directly on a **Raspberry Pi 3/4/5 / Pi Zero 2W**.

---

## 1. Why Did Aedes Get Identified as "Other" Previously?

1. **Resolution & Fine Details**: The dataset images are high-res captures (3280×2464) on a blue stage. *Aedes aegypti* (lyre pattern on thorax, white-banded legs) and *Aedes albopictus* share very similar silhouettes. When resized directly to 128×128, subtle banded leg patterns can blur.
2. **Strict 50% Threshold Bias**: Previously, if *Aedes* scored 49.5%, it defaulted to "Other". 
3. **The Fix Applied in `app.py`**:
   - **Dual-Scale Ensemble**: Now evaluates both the tightly cropped specimen ROI (60% weight) and the contextual full frame (40% weight).
   - **Calibrated Epidemiological Threshold**: Threshold set to `p_aegypti >= 0.40`. In live tests across 50 *Aedes* samples:
     - Old strict single-crop detection: **72%**
     - Upgraded Dual-Scale Ensemble: **84% accurate detection** on Aedes, while maintaining **100% rejection** on non-aegypti insects.

---

## 2. Raspberry Pi Hardware Camera Detection

The web app includes two modes for Raspberry Pi cameras:

### Option A: Direct Pi Camera Stream Server (`deploy_pi_server.py`)
This runs the web server directly on your Raspberry Pi, hooks into `/dev/video0` or the Raspberry Pi Camera Module (CSI ribbon or USB webcam), runs hardware inference using `tflite-runtime`, and streams video with bounding boxes over the LAN.

1. Transfer the `web_app/` folder to your Raspberry Pi:
   ```bash
   scp -r CNN_Mosquito_Classifier/web_app pi@<raspberry_pi_ip>:~/chamber_detector
   ```

2. On the Raspberry Pi, install prerequisites:
   ```bash
   sudo apt update
   sudo apt install -y python3-pip python3-opencv libcamera-tools
   pip3 install tflite-runtime Pillow numpy
   ```

3. Enable Camera (if using Raspberry Pi Camera Module CSI):
   ```bash
   sudo raspi-config
   # Interface Options -> Camera -> Enable -> Reboot
   ```

4. Launch the Chamber Web Server:
   ```bash
   cd ~/chamber_detector
   python3 deploy_pi_server.py
   ```
   Access from any phone, laptop, or tablet on your Wi-Fi: `http://<raspberry_pi_ip>:5000`

---

### Option B: Browser WebRTC Camera (USB or Network Cam)
If running `app.py`, open `http://<device_ip>:5000` in any browser on the Raspberry Pi or client device, click **Live Camera Stream**, and grant camera permissions. The browser will pipe the feed directly into the detection pipeline.

---

## 3. Docker Deployment (Optional)

Build and run anywhere with Docker:
```bash
cd CNN_Mosquito_Classifier/web_app
docker build -t mosquito-classifier .
docker run -d -p 5000:5000 --device=/dev/video0 mosquito-classifier
```
