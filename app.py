from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from itenaryGenerator import ItinerarySystem
from models.itinerary import ItineraryModel
from bson import ObjectId
import uvicorn
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

app = FastAPI(title="Travel Itinerary API")
itinerary_model = ItineraryModel()

# Define specific localhost origins (useful for predefined cases)
allowed_localhosts = [
    "http://localhost:3100",
]

# Add middleware dynamically
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_localhosts,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PreferencesInput(BaseModel):
    destination: str
    budget: float
    interests: List[str]
    accessibility_needs: Optional[List[str]] = []
    preferred_times: Optional[Dict[str, str]] = {
        "start_time": "10:00",
        "end_time": "20:00",
    }
    origin: Optional[str] = "New York"
    start_date: str
    end_date: str


def generate_itinerary_background(job_id, preferences):
    try:
        system = ItinerarySystem(preferences)
        result = system.create_itinerary()
        itinerary_model.update_job(ObjectId(job_id), "completed", result=result)
    except Exception as e:
        itinerary_model.update_job(ObjectId(job_id), "failed", error=str(e))


@app.post("/generate-itinerary")
async def generate_itinerary(
    preferences: PreferencesInput, background_tasks: BackgroundTasks
):
    try:
        # Convert Pydantic model to dict
        preferences_dict = preferences.dict()

        # Create job and start background task
        job_id = itinerary_model.create_job(preferences_dict)
        background_tasks.add_task(
            generate_itinerary_background, job_id, preferences_dict
        )

        return {"job_id": str(job_id)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/itineraries")
async def get_itineraries():
    try:
        jobs = itinerary_model.get_all_jobs()
        return {"itineraries": jobs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/itinerary/{job_id}")
async def get_itinerary(job_id: str):
    try:
        job = itinerary_model.get_job(ObjectId(job_id))
        if not job:
            raise HTTPException(status_code=404, detail="Itinerary not found")
        return job
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/itinerary/{job_id}")
async def delete_itinerary(job_id: str):
    try:
        success = itinerary_model.delete_job(ObjectId(job_id))
        if not success:
            raise HTTPException(status_code=404, detail="Itinerary not found")
        return {"message": "Itinerary deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"message": "Welcome to the Travel Itinerary API"}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT")), reload=True)
