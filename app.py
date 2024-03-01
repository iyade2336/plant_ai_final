from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
from io import BytesIO
from torchvision.models import ResNet18_Weights
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

app = FastAPI()

# قائمة الأصول المسموح بها
origins = [
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <html>
        <head>
            <title>تطبيق FastAPI</title>
        </head>
        <body>
            <h1>مرحبًا بك في تطبيق FastAPI!</h1>
            <p>هذه صفحة الترحيب.</p>
        </body>
    </html>
    """

class_names = ['apple scab', 'apple black rot', 'cedar apple rust', 'apple healthy', 'blueberry healthy', 'cherry powdery mildew', 'cherry healthy', 'corn cercospora leaf gray leaf spot', 'corn common rust', 'corn northern leaf blight', 'corn healthy', 'grape black rot', 'grape black measles', 'grape leaf blight', 'grape healthy', 'orange haunglongbing', 'peach bacterial spot', 'peach healthy', 'pepper bell Bacterial spot', 'pepper bell healthy', 'potato early blight', 'potato late blight', 'potato healthy', 'raspberry healthy', 'soybean healthy', 'squash powdery mildew', 'strawberry leaf scotch', 'strawberry healthy', 'tomato bacterial spot', 'tomato early blight', 'tomato late blight', 'tomato leaf mold', 'tomato septoria leaf spot', 'tomato spider mites two spotted spider mite', 'tomato target spot', 'Tomato yellow leaf curl virus', 'tomato mosaic virus', 'tomato healthy']

model_path = 'model.pth'
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

def transform_image(image_bytes):

    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    image_resized = cv2.resize(image, (256, 256))
    image_center_crop = image_resized[16:240, 16:240]
    image_normalized = image_center_crop / 255.0
    image_rgb = cv2.cvtColor(image_normalized, cv2.COLOR_BGR2RGB)
    image_transposed = np.transpose(image_rgb, (2, 0, 1))
    image_tensor = torch.tensor(image_transposed).unsqueeze(0).float()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image_normalized = normalize(image_tensor)
    return image_normalized

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    image_tensor = transform_image(image_data)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        max_prob, predicted = torch.max(probabilities, 1)
    
    predicted_class = class_names[predicted.item()]
    confidence = max_prob.item()

    # تحقق من قيمة الثقة
    if confidence < 0.5:  # يمكنك تعديل العتبة حسب الحاجة
        return JSONResponse(content={"error": "غير قادر على التعرف على نوع النبتة بثقة."})
    
    return JSONResponse(content={"predicted_class": predicted_class, "confidence": confidence})
