from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import os

from src.rag_engine import generate_rag_itinerary

load_dotenv()

app = FastAPI()

class ItineraryRequest(BaseModel):
    city: str
    trip_duration: str = "1-day"
    budget: str = "Moderate"
    interests: list[str]
    location: str | None = None


@app.get("/")
def root():
    return {"status": "RAG server running!"}


@app.post("/generate_itinerary")
def generate_itinerary(req: ItineraryRequest):

    # DIRECT RAG CALL → retrieves monuments + builds prompt + calls Groq
    itinerary_text = generate_rag_itinerary(
        city=req.city,
        location=req.location,
        trip_duration=req.trip_duration,
        budget=req.budget,
        interests=req.interests
    )

    return {"itinerary": itinerary_text}
