from __future__ import annotations

from flask import Flask, redirect

from app.driving.api.routes import api_bp as driving_bp
from app.tuning.routes import tuning_bp


def create_app() -> Flask:
    app = Flask(__name__)
    app.register_blueprint(driving_bp)
    app.register_blueprint(tuning_bp)

    @app.get("/")
    def root_redirect():
        return redirect("/driving/")

    return app
