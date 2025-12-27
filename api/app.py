from fastapi import FastAPI
from api.configs.db import init_db
from api.route.user_route import router as user_router
from api.route.telegram_route import router as telegram_router


def create_app() -> FastAPI:
    app = FastAPI(title="Shoplifting API")

    @app.on_event("startup")
    def _startup():
        # initialize DB (creates tables)
        init_db()

    app.include_router(user_router)
    app.include_router(telegram_router)
    return app


app = create_app()
