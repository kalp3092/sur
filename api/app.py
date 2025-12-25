from fastapi import FastAPI
from api.configs.db import init_db
from api.route.user_route import router as user_router


def create_app() -> FastAPI:
    app = FastAPI(title="Shoplifting API")

    @app.on_event("startup")
    def _startup():
        # initialize DB (creates tables)
        init_db()

    app.include_router(user_router)
    return app


app = create_app()
