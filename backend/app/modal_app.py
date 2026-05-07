import modal

from backend.app.main import app as fastapi_app

app = modal.App("my-ai-app")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "fastapi",
        "uvicorn",
        "transformers",
        "torch"
    )
)


@app.function(image=image)
@modal.asgi_app()
def api():
    return fastapi_app