<div align="center">

# ✈️ Flight Delay Predictor

**Will your flight land 15+ minutes late? A machine-learning app trained on 40+ million US DOT flight records.**

[![Live Demo](https://img.shields.io/badge/Live_Demo-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://flight-delay-predictor-87mn.onrender.com)
[![CI](https://github.com/arpitk2509/flight-delay-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/arpitk2509/flight-delay-predictor/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.7-2E7D32)
![FastAPI](https://img.shields.io/badge/FastAPI-0.143-009688?logo=fastapi&logoColor=white)
![DuckDB](https://img.shields.io/badge/DuckDB-Parquet-FFF000?logo=duckdb&logoColor=black)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)

[**Try the live app**](https://flight-delay-predictor-87mn.onrender.com) · [How it works](#-how-it-works) · [Results](#-results) · [Run it locally](#-run-it-locally) · [API](#-api)

<br>

<img src="docs/screenshot.png" alt="Flight Delay Predictor web app" width="860">

<sub>The free server sleeps when idle, so the first load can take up to a minute.</sub>

</div>

---

## Contents

- [What it does](#-what-it-does)
- [How it works](#-how-it-works)
- [Results](#-results)
- [Key design decisions](#-key-design-decisions)
- [Tech stack](#-tech-stack)
- [Run it locally](#-run-it-locally)
- [Reproduce the training](#-reproduce-the-training)
- [API](#-api)
- [Deployment](#-deployment)
- [Project structure](#-project-structure)
- [Limitations and roadmap](#-limitations-and-roadmap)
- [Data and credits](#-data-and-credits)

## 🎯 What it does

Pick a US domestic route, an airline, a date and a scheduled departure time. The app returns:

- the **probability** that the flight arrives **15 or more minutes late** (the official DOT definition of "delayed"),
- a **risk band** (low, medium, high) relative to a typical flight,
- the **top reasons** behind the estimate, such as departure hour, last year's delay rate at that airport, or the airline's record,
- a chart of **late-arrival rate by departure hour** at the origin airport.

Everything is computed from information known **before departure**, so the estimate is something a passenger could really use when booking.

<br>

<img src="docs/prediction_breakdown.png" alt="Flight Delay Predictor feature breakdown and hourly chart" width="860">

<br>

## 🧭 How it works

```mermaid
flowchart LR
    A[BTS monthly CSV zips<br/>2018 to 2024] --> B[Parquet<br/>ZSTD compressed]
    B --> C[DuckDB SQL<br/>features and history rates]
    C --> D[LightGBM<br/>time-based split]
    D --> E[Isotonic calibration]
    E --> F[(artifacts/<br/>model + lookups)]
    F --> G[FastAPI<br/>/api/predict]
    G --> H[Web page<br/>HTML, CSS, JS]
    H -. same origin .-> G
    G --> I[Docker on Render]
