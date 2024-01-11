import openai
from fastapi import FastAPI, Form, HTTPException
from starlette.responses import HTMLResponse
from starlette.requests import Request
from fastapi.templating import Jinja2Templates


app = FastAPI()

openai.api_key = "your-api-key"

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index_fastapi.html", {"request": request, "answer": None})

@app.post("/ask", response_class=HTMLResponse)
async def get_answer(request: Request, question: str = Form(...)):
    try:
        response = openai.Completion.create(
            engine="gpt-3.5-turbo-instruct",
            prompt=question,
            max_tokens=150
        )
        answer = response.choices[0].text.strip()
        return templates.TemplateResponse("index.html", {"request": request, "answer": answer})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
