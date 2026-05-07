import modal

from app.main import app as fastapi_app

app = modal.App("my-ai-app")

image = (
    modal.Image.debian_slim()
    .apt_install("packages.txt")
    .pip_install_from_requirements("requirements.txt")
)

@app.function(image=image)
@modal.asgi_app()
def api():
    return fastapi_app