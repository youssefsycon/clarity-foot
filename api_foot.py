import requests
import uvicorn as uvicorn
from fastapi import FastAPI
import json

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/getmatch")
async def predict():
    uri = 'http://api.football-data.org/v4/areas/'
    headers = {'X-Auth-Token': '6d833c3caa574789a94031822e99949b'}

    response = requests.get(uri, headers=headers)
    return response.json()['areas']


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)
