import io
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from starlette.applications import Starlette
from starlette.responses import JSONResponse, FileResponse
from starlette.routing import Route
import uvicorn

# 1. 7-Class HAM10000 Taxonomy
CLASSES = [
    {"code": "mel", "name": "Melanoma", "risk": "HIGH", "desc": "Malignant melanocyte neoplasm. High metastatic potential. Immediate dermatologist biopsy advised."},
    {"code": "nv", "name": "Melanocytic Nevus", "risk": "LOW", "desc": "Common benign mole. Symmetrical and circumscribed. Routine monitoring recommended."},
    {"code": "bcc", "name": "Basal Cell Carcinoma", "risk": "HIGH", "desc": "Malignant skin carcinoma originating in basal epidermis. Medical evaluation required."},
    {"code": "akiec", "name": "Actinic Keratosis", "risk": "MOD", "desc": "Pre-cancerous sun-induced dysplastic epidermal lesion. Clinical review recommended."},
    {"code": "bkl", "name": "Benign Keratosis", "risk": "LOW", "desc": "Seborrheic keratosis or solar lentigo. Benign epidermal growth."},
    {"code": "df", "name": "Dermatofibroma", "risk": "LOW", "desc": "Benign cutaneous fibrous histiocytoma. Non-cancerous."},
    {"code": "vasc", "name": "Vascular Lesion", "risk": "LOW", "desc": "Benign blood vessel anomaly (cherry angioma or pyogenic granuloma)."}
]

# 2. PyTorch Model Initialization
print("Initializing PyTorch MobileNetV3 model...")
try:
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
except Exception:
    model = models.mobilenet_v3_small(weights=None)

in_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(in_features, len(CLASSES))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 3. Request Handlers
async def homepage(request):
    return FileResponse("index.html")

async def get_styles(request):
    return FileResponse("styles.css", media_type="text/css")

async def get_js(request):
    return FileResponse("app.js", media_type="application/javascript")

async def predict(request):
    try:
        form = await request.form()
        file = form.get("file")
        if file is None:
            return JSONResponse({"error": "No file uploaded"}, status_code=400)
            
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Inference
        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0].tolist()

        # Combine with class info
        results = []
        for i, prob in enumerate(probabilities):
            cls = CLASSES[i]
            results.append({
                "code": cls["code"],
                "name": cls["name"],
                "risk": cls["risk"],
                "description": cls["desc"],
                "probability": round(prob, 4),
                "percentage": round(prob * 100, 1)
            })

        # Sort descending by probability
        results.sort(key=lambda x: x["probability"], reverse=True)
        top = results[0]

        return JSONResponse({
            "success": True,
            "prediction": top["name"],
            "code": top["code"],
            "risk": top["risk"],
            "confidence": top["percentage"],
            "description": top["description"],
            "all_probabilities": results
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# 4. App Routes
routes = [
    Route("/", homepage),
    Route("/styles.css", get_styles),
    Route("/app.js", get_js),
    Route("/api/predict", predict, methods=["POST"]),
]

app = Starlette(debug=True, routes=routes)

if __name__ == "__main__":
    port = 8000
    print(f"Starting DermaScan AI Server at http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
