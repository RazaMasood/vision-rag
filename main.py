from fastapi import FastAPI
from api.routers import uploadfile, query

app = FastAPI(
    title="RAG Document Processing API",
    description="Upload documents and query them using RAG",
    version="1.0.0"
)

# Include routers
app.include_router(uploadfile.router, prefix="/api/v1")
app.include_router(query.router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "RAG Document Processing API", "version": "1.0.0"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)