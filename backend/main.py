from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import FastAPI, status

from router import llmapi, llmfitapi

app = FastAPI(title='Local-AI')

origins = [
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "http://localhost:8001",
    "http://127.0.0.1:8001",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(llmapi.router)
app.include_router(llmfitapi.router)

@app.get("/", status_code=status.HTTP_200_OK)
def health() -> JSONResponse:
    return {"message":"Server is Healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app=app, host="127.0.0.1", port="8000")
