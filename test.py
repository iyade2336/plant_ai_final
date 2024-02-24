import requests

url = 'http://127.0.0.1:8000/predict/'
files = {'file': open('C:/Users/iyade/OneDrive/Desktop/big_project-main/big_project-main/class 5.JPG', 'rb')}  # تأكد من تحديث المسار إلى مسار الصورة التي تريد تصنيفها

response = requests.post(url, files=files)
print(response.json())
