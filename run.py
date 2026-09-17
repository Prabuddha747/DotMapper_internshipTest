"""Convenience entrypoint: `python run.py`. Equivalent to `uvicorn app.main:app --reload`."""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.main:app", reload=True)
