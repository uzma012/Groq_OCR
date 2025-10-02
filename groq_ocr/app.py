import numpy as np
from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
import pytesseract
from providers import GroqProvider  # Import GroqProvider
from prompts import SYSTEM_PROMPT, USER_PROMPT  # Import your prompts
from dotenv import load_dotenv
import json

load_dotenv()

from utils import perform_ocr

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# Initialize GroqProvider
groq_provider = GroqProvider()
GROQ_MODEL = "llama-3.3-70b-versatile"  # Or your preferred Groq model

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Welcome to the OCR API"}


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/ocr/")
async def ocr_receipt(file: UploadFile):
    # Check if the uploaded file is an image
    if file.content_type.startswith("image"):
        image_bytes = await file.read()
        img_array = np.frombuffer(image_bytes, np.uint8)

        ocr_text = perform_ocr(img_array)
        print("OCR Text:", ocr_text)
        # Pass OCR text to Groq for parsing
        result_json = groq_provider.get_response(
            ocr_text=ocr_text,
            json_schema={},  # Or your schema if needed
            model=GROQ_MODEL
        )
        # Parse the JSON string to a Python dict
        try:
            result_dict = json.loads(result_json)
        except Exception:
            result_dict = {"error": "Failed to parse Groq response", "raw": result_json}
        return JSONResponse(content=result_dict, status_code=200)
    else:
        return {"error": "Uploaded file is not an image"}