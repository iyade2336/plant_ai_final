from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

app = FastAPI()

# لخدمة ملفات CSS والموارد الثابتة
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_index():
    return Path("templates/index.html").read_text()

# الكود الخاص بتحميل الصورة والتنبؤ هنا
