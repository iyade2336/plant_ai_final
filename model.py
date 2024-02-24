import torch
from torchvision import models

# افترض أننا نستخدم نموذج resnet18 كمثال
model = models.resnet18(pretrained=True)  # إنشاء نموذج، تعطيل pretrained إذا كنت تحمّل حالة نموذج مخصصة

# تحميل حالة النموذج
model_path = 'model.pth'
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

model.eval()  # تعيين النموذج في وضع التقييم
