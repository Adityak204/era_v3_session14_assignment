from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import httpx
import os

app = FastAPI()
# app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# @app.post("/generate")
# async def generate_text(text: str, seq_length: int):
#     async with httpx.AsyncClient() as client:
#         response = await client.post(
#             "http://backend:8000/predict", json={"text": text, "seq_length": seq_length}
#         )
#         return response.json()


@app.post("/generate")
async def generate_text(request: Request):
    # Get JSON data from request
    data = await request.json()
    print("User Input: ", data)
    text = data.get("text")
    seq_length = data.get("seq_length")

    async with httpx.AsyncClient(timeout=50.0) as client:
        response = await client.post(
            "http://backend:8000/predict", json={"text": text, "seq_length": seq_length}
        )
        return response.json()
