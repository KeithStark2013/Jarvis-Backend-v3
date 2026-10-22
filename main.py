import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq

app = FastAPI(title="J.A.R.V.I.S. Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def home():
    return {
        "status": "online",
        "assistant": "J.A.R.V.I.S.",
        "version": "3.0",
    }


@app.post("/chat")
def chat(request: ChatRequest):
    message = request.message.strip()

    if not message or len(message) > 2000:
        raise HTTPException(
            status_code=400,
            detail="Message must be between 1 and 2000 characters.",
        )

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="AI key has not been configured.",
        )

    try:
        client = Groq(api_key=api_key)

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are J.A.R.V.I.S., a composed, "
                        "intelligent personal assistant. "
                        "Use polished British English, speak "
                        "concisely, and address the user as sir "
                        "when natural. Be honest about your abilities."
                    ),
                },
                {"role": "user", "content": message},
            ],
        )

        answer = response.choices[0].message.content
        return {"reply": answer or "At your service, sir."}

    except Exception:
        raise HTTPException(
            status_code=502,
            detail="The AI service could not complete the request.",
        )
