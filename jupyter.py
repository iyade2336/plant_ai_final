import requests
from ipywidgets import FileUpload, Button, Output
from IPython.display import display, clear_output

# إعداد واجهة التحميل
upload = FileUpload(accept='image/*', multiple=False)
upload_btn = Button(description="تحميل الصورة")
output = Output()

display(upload, upload_btn, output)

def on_upload_clicked(b):
    with output:
        clear_output()
        if not upload.data:
            print("الرجاء تحميل صورة.")
            return

        files = {'file': ('image.jpg', upload.data[-1], 'image/jpeg')}
        response = requests.post('http://127.0.0.1:8000/predict/', files=files)
        if response.status_code == 200:
            result = response.json()
            print(f"نتيجة التنبؤ: {result['predicted_class']}")
        else:
            print("حدث خطأ أثناء التنبؤ.")

upload_btn.on_click(on_upload_clicked)
