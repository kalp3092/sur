"""Application entrypoint to run FastAPI server and detector."""

import uvicorn
from src.config import get_settings


def run():
    s = get_settings()
    uvicorn.run("src.api:app", host=s.host, port=s.port, reload=False)


if __name__ == '__main__':
    run()
