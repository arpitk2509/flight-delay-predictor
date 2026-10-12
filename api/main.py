"""
FastAPI backend. Serves:
    /api/*   -> JSON endpoints used by the web page
    /        -> the web page itself (the ./web folder)
Because both come from the same server, the browser needs no CORS setup.

Run locally:  uvicorn api.main:app --reload --port 8000
"""
import datetime as dt
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.predictor import Predictor, RouteNotFound

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["predictor"] = Predictor()  # load the model ONCE at startup, not per request
    yield
    state.clear()


app = FastAPI(title="Flight Delay Predictor", version="1.0.0", lifespan=lifespan)

# Only needed if you host the web page on a DIFFERENT domain than the API.
origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
if origins:
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])


class FlightRequest(BaseModel):
    flight_date: dt.date
    dep_time: str = Field(pattern=r"^([01]\d|2[0-3]):[0-5]\d$", examples=["18:45"])
    carrier: str = Field(min_length=2, max_length=3)
    origin: str = Field(min_length=3, max_length=3)
    dest: str = Field(min_length=3, max_length=3)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/meta")
def meta():
    p: Predictor = state["predictor"]
    return {**p.meta(), "airports": p.origins()}


@app.get("/api/destinations")
def destinations(origin: str = Query(min_length=3, max_length=3)):
    return state["predictor"].destinations(origin.upper())


@app.get("/api/carriers")
def carriers(origin: str = Query(min_length=3, max_length=3), dest: str = Query(min_length=3, max_length=3)):
    return state["predictor"].carriers(origin.upper(), dest.upper())


@app.post("/api/predict")
def predict(req: FlightRequest):
    try:
        return state["predictor"].predict(
            req.flight_date, req.dep_time, req.carrier.upper(), req.origin.upper(), req.dest.upper())
    except RouteNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))


# Must be mounted LAST so it doesn't shadow the /api routes.
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
