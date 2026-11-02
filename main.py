import os
import json
import urllib.request
import urllib.parse

from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, root_validator
from groq import Groq


app = FastAPI(
    title="J.A.R.V.I.S. Backend",
    version="4.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------
# REQUEST COMPATIBILITY
# ----------------------------------

class ChatRequest(BaseModel):
    message: str

    @root_validator(pre=True)
    def accept_frontend_formats(cls, values):
        if isinstance(values, str):
            return {"message": values}

        if isinstance(values, dict):
            if "message" not in values:
                for field in ("text", "prompt", "input"):
                    if field in values:
                        values["message"] = values[field]
                        break

        return values


# ----------------------------------
# LOCATION DATA
# ----------------------------------

LOCATIONS = {
    "kuala lumpur": {
        "latitude": 3.1390,
        "longitude": 101.6869
    },
    "kl": {
        "latitude": 3.1390,
        "longitude": 101.6869
    },
    "shah alam": {
        "latitude": 3.0733,
        "longitude": 101.5185
    }
}


def malaysia_time():
    return datetime.now(
        ZoneInfo("Asia/Kuala_Lumpur")
    ).strftime("%A, %d %B %Y, %I:%M:%S %p")


# ----------------------------------
# LIVE AQI SERVICE
# ----------------------------------

def get_aqi(city):
    city = city.lower().strip()

    if city not in LOCATIONS:
        raise HTTPException(
            status_code=400,
            detail="Supported locations: Kuala Lumpur and Shah Alam."
        )

    location = LOCATIONS[city]

    params = urllib.parse.urlencode({
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "current": "us_aqi,pm2_5",
        "timezone": "Asia/Kuala_Lumpur"
    })

    url = (
        "https://air-quality-api.open-meteo.com/v1/air-quality?"
        + params
    )

    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "JARVIS-Assistant/4.0"}
        )

        with urllib.request.urlopen(
            request,
            timeout=12
        ) as response:
            data = json.loads(response.read().decode("utf-8"))

        current = data.get("current", {})

        return {
            "city": city.title(),
            "us_aqi": current.get("us_aqi"),
            "pm2_5": current.get("pm2_5"),
            "updated": current.get("time"),
            "source": "Open-Meteo Air Quality API"
        }

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=502,
            detail="The AQI service is temporarily unavailable."
        )


# ----------------------------------
# API ROUTES
# ----------------------------------

@app.get("/")
def home():
    return {
        "status": "online",
        "assistant": "J.A.R.V.I.S.",
        "version": "4.0"
    }


@app.get("/time")
def get_time():
    return {
        "time": malaysia_time(),
        "timezone": "Asia/Kuala_Lumpur"
    }


@app.get("/aqi")
def aqi(city: str = "Kuala Lumpur"):
    return get_aqi(city)


# ----------------------------------
# CHAT ENGINE
# ----------------------------------

@app.post("/chat")
def chat(request: ChatRequest):

    message = request.message.strip()

    if not message or len(message) > 2000:
        raise HTTPException(
            status_code=400,
            detail="Message must be between 1 and 2000 characters."
        )

    lower_message = message.lower()

    # Malaysia time requests
    time_words = (
        "what time",
        "current time",
        "check time",
        "what's the time",
        "date and time"
    )

    if any(word in lower_message for word in time_words):
        return {
            "reply": (
                f"The current time in Malaysia is "
                f"{malaysia_time()}, sir."
            )
        }

    # AQI requests
    if "aqi" in lower_message or "air quality" in lower_message:

        city = "kuala lumpur"

        if "shah alam" in lower_message:
            city = "shah alam"

        result = get_aqi(city)

        aqi_value = result["us_aqi"]
        pm_value = result["pm2_5"]

        if aqi_value is None:
            reply = (
                f"I could not obtain a current AQI reading "
                f"for {result['city']}, sir."
            )
        else:
            reply = (
                f"The latest available US AQI reading for "
                f"{result['city']} is {aqi_value}. "
                f"PM2.5 is {pm_value} µg/m³. "
                f"The reading was updated at "
                f"{result['updated']}. "
                f"Source: {result['source']}."
            )

        return {
            "reply": reply,
            "aqi_data": result
        }

    # Groq AI
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="AI key has not been configured."
        )

    try:
        client = Groq(api_key=api_key)

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are J.A.R.V.I.S., a highly capable "
                        "personal AI assistant. "
                        "Use polished British English. "
                        "Be composed, analytical and concise. "
                        "Address the user as sir when natural. "
                        "Never claim to have performed an action "
                        "unless it was actually completed. "
                        "Never claim access to devices or accounts "
                        "without an authorised connection. "
                        "Ask for permission before sensitive actions."
                    )
                },
                {
                    "role": "user",
                    "content": message
                }
            ],
        )

        answer = response.choices[0].message.content

        return {
            "reply": answer or "At your service, sir.",
            "assistant": "J.A.R.V.I.S.",
            "status": "success"
        }

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=502,
            detail="The AI service could not complete the request."
        )
