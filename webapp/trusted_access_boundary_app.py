from __future__ import annotations

from flask import Flask, Response


def create_trusted_access_boundary_app() -> Flask:
    """Create the dedicated trusted-access boundary application only."""
    app = Flask(__name__, static_folder=None)

    @app.after_request
    def _no_store(response):
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.route("/", methods=["GET"])
    def trusted_access_boundary_root():
        return Response("ok\n", status=200, mimetype="text/plain")

    @app.route("/auth/certificate/verify", methods=["GET"])
    def auth_certificate_verify():
        from webapp.parser.auth.trusted_access_boundary import (
            verify_trusted_certificate_access,
        )

        return verify_trusted_certificate_access()

    return app


app = create_trusted_access_boundary_app()
