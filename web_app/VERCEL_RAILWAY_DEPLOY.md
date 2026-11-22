# Vercel (Frontend) + Railway (Backend) Deployment Guide

This architecture gives you:
* **Frontend on Vercel**: Global Edge CDN, zero latency, auto HTTPS, and high availability.
* **Backend on Railway**: Full Python container with OpenCV, TensorFlow/TFLite, running the dual-scale detection engine.

---

## Step 1: Deploy Backend to Railway

1. Push your project or the `CNN_Mosquito_Classifier/web_app` folder to a GitHub repository.
2. Go to [railway.app](https://railway.app/) and sign in.
3. Click **New Project** → **Deploy from GitHub repo** → select your repo.
4. If your app is inside a subfolder, in Railway go to **Settings** → **Root Directory** and set it to:
   ```
   CNN_Mosquito_Classifier/web_app
   ```
5. Railway will automatically detect the [`Dockerfile`](file:///C:/Users/johnl/OneDrive/Desktop/TRAIN/CNN_Mosquito_Classifier/web_app/Dockerfile) and [`railway.toml`](file:///C:/Users/johnl/OneDrive/Desktop/TRAIN/CNN_Mosquito_Classifier/web_app/railway.toml) and build the container.
6. Once deployed, go to **Settings** → **Networking** → click **Generate Domain** (e.g. `https://mosquito-classifier-production.up.railway.app`).
7. Test the healthcheck endpoint: `https://<your-railway-domain>/api/status`. It should respond with:
   ```json
   {"status": "online", "model_architecture": "MobileNetV2 TFLite", "model_accuracy": "86.27% balanced"}
   ```

---

## Step 2: Deploy Frontend to Vercel

1. Go to [vercel.com](https://vercel.com/) and sign in.
2. Click **Add New** → **Project** → import your GitHub repo.
3. Under **Root Directory**, click edit and select:
   ```
   CNN_Mosquito_Classifier/web_app
   ```
4. Framework Preset: Leave as **Other**.
5. Click **Deploy**. Vercel will deploy [`index.html`](file:///C:/Users/johnl/OneDrive/Desktop/TRAIN/CNN_Mosquito_Classifier/web_app/index.html) and apply routing from [`vercel.json`](file:///C:/Users/johnl/OneDrive/Desktop/TRAIN/CNN_Mosquito_Classifier/web_app/vercel.json).

---

## Step 3: Link Vercel Frontend to Railway Backend

Your frontend automatically listens for your Railway URL. You can set it in two easy ways:

### Method A (No Redeployment Needed):
Open your Vercel site in your browser, press `F12` (Developer Console), and run:
```javascript
localStorage.setItem('API_BASE_URL', 'https://your-railway-app.up.railway.app');
location.reload();
```

### Method B (Permanent in HTML):
In [`index.html`](file:///C:/Users/johnl/OneDrive/Desktop/TRAIN/CNN_Mosquito_Classifier/web_app/index.html), line 527:
```javascript
const API_BASE = 'https://your-railway-app.up.railway.app';
```
Commit and push. Vercel will automatically redeploy!

---

## Step 4: Connecting the Raspberry Pi Camera to the Cloud Deployment

When using the Vercel + Railway deployed system:
1. Open your Vercel site on any browser, laptop, or on the Raspberry Pi itself:
   `https://your-app.vercel.app`
2. Under **Live Camera Stream**, select the connected camera. The browser will capture frames from your camera hardware and send frames directly to the Railway backend for inference.
3. Bounding boxes (RED for *Aedes aegypti*, BLUE for Other) render in real time on the video stream.
