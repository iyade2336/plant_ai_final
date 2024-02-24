from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
from torchvision import models, transforms
from PIL import Image
from io import BytesIO
from torchvision.models import ResNet18_Weights
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# خدمة الملفات الثابتة من مجلد 'web'
# app.mount("/web", StaticFiles(directory="web"), name="web")


# قائمة الأصول المسموح بها
origins = [
    "http://localhost:5500",  # ضع هنا عنوان URL الخاص بصفحة الويب الخاصة بك
    "http://127.0.0.1:5500",
    # يمكنك إضافة أصول أخرى حسب الحاجة
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # يمكنك استخدام ["*"] للسماح بجميع الأصول (ليس موصى به للإنتاج)
    allow_credentials=True,
    allow_methods=["*"],  # يسمح بجميع الطرق (GET, POST, ...)
    allow_headers=["*"],  # يسمح بجميع الرؤوس
)

# تعريف مسارات API الخاصة بك هنا


# قائمة بأسماء الفئات
class_names = ['apple scab','apple black rot','cedar apple rust','apple healthy','blueberry healthy','cherry powdery mildew','cherry healthy','corn cercospora leaf gray leaf spot','corn common rust','corn northern leaf blight','corn healthy','grape black rot','grape black measles','grape leaf blight','grape healthy','orange haunglongbing', 'peach bacterial spot','peach healthy','pepper bell Bacterial spot','pepper bell healthy','potato early blight','potato late blight','potato healthy','raspberry healthy','soybean healthy','squash powdery mildew','strawberry leaf scotch','strawberry healthy','tomato bacterial spot','tomato early blight','tomato late blight','tomato leaf mold','tomato septoria leaf spot','tomato spider mites two spotted spider mite', 'tomato target spot','Tomato yellow leaf curl virus','tomato mosaic virus','tomato healthy']

# تعيين مسار نموذج حالة النموذج المدرب
model_path = 'model.pth'

# إنشاء النموذج وتعديل الطبقة الأخيرة
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))

# تحميل حالة النموذج المدرب
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()  # تعيين النموذج في وضع التقييم


# تعريف تحويلات الصورة
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # قراءة الصورة وتحويلها
    image_data = await file.read()
    image = Image.open(BytesIO(image_data))
    image = transform(image)
    image = image.unsqueeze(0)  # إضافة بُعد الدُفعات

    # التنبؤ
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    
    # الحصول على اسم الفئة المتنبأ بها من القائمة
    predicted_class = class_names[predicted.item()]

    return JSONResponse(content={"predicted_class": predicted_class})
