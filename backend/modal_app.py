import modal

from app.main import app as fastapi_app

app = modal.App("my-ai-app")

def get_apt_packages():
    with open("packages.txt", "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

image = (
    modal.Image.debian_slim()
    .apt_install(*get_apt_packages())
    .pip_install_from_requirements("requirements.txt")
    .add_local_python_source("app")
    .add_local_python_source("packages.txt")
    .add_local_python_source("requirements.txt")
)

@app.function(image=image)
@modal.asgi_app()
def api():
    return fastapi_app