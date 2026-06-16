import os
import traceback
import requests
import base64
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# En la nube no dejamos la clave escrita; la leerá el sistema de forma segura
API_KEY = os.getenv("GEMINI_API_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/buscar-objeto")
async def buscar_objeto(
    foto: UploadFile = File(...),
    objeto_buscado: str = Form(...)
):
    try:
        contents = await foto.read()
        image_base64 = base64.b64encode(contents).decode('utf-8')

        prompt = f"""
        Tu único objetivo es localizar de manera exacta el siguiente objeto en la imagen: "{objeto_buscado}".
        Devuelve las coordenadas utilizando una caja delimitadora 2D (bounding box) en formato [ymin, xmin, ymax, xmax].
        Los valores deben estar normalizados en un rango de 0 a 1000 basándote en la resolución de la imagen.
        Si el objeto no está presente o no se ve en absoluto, marca 'encontrado' como false.

        Devuelve la respuesta estrictamente en este formato JSON:
        {{
            "encontrado": true o false,
            "box_2d": [ymin, xmin, ymax, xmax],
            "etiqueta": "{objeto_buscado}"
        }}
        """

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={API_KEY}"

        payload = {
            "contents": [{"parts": [{"text": prompt}, {"inlineData": {"mimeType": foto.content_type, "data": image_base64}}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0.1}
        }

        response = requests.post(url, json=payload)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)

        return response.json()['candidates'][0]['content']['parts'][0]['text']

    except Exception as e:
        print("--- ERROR ---")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
