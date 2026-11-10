# DermaScan AI — Clinical Skin Disease Classifier & Screening Studio

An end-to-end, high-performance dermatological screening application powered by a custom **Medical UI Dashboard** and a **PyTorch MobileNetV3 Deep Learning Engine** using the **HAM10000** taxonomy.

---

## 🌟 Key Features
- **Custom High-Fidelity UI:** Replaced generic Streamlit with a bespoke, responsive clinical interface featuring dark glassmorphism, glowing risk indicators, and laser scan animations.
- **7-Class HAM10000 Diagnostic Taxonomy:**
  1. `mel` — **Melanoma** (Malignant, High Urgency)
  2. `bcc` — **Basal Cell Carcinoma** (Malignant, High Urgency)
  3. `akiec` — **Actinic Keratosis** (Precancerous, Moderate Urgency)
  4. `bkl` — **Benign Keratosis** (Benign Epidermal Proliferation)
  5. `nv` — **Melanocytic Nevus** (Common Harmless Mole)
  6. `df` — **Dermatofibroma** (Benign Cutaneous Histiocytoma)
  7. `vasc` — **Vascular Lesion** (Benign Blood Vessel Anomaly)
- **Interactive Ingestion:**
  - Drag-and-drop file upload (JPEG, PNG).
  - Real-time device webcam capture.
  - 1-click test presets (Melanoma, Basal Cell, Benign Mole, Keratosis).
- **Differential Class Probabilities:** Real-time probability bars for all 7 disease categories.
- **Printable Medical Report:** Instant export to clinical PDF report (`window.print()`).

---

## 🚀 How to Run

### Method 1: Double-Click Launcher (Easiest)
Simply double-click **`run_app.bat`**. It launches the PyTorch server and automatically opens `http://localhost:8000` in your browser.

### Method 2: Command Line
```powershell
.venv\Scripts\activate
python server.py
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser.
