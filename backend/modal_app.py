import modal
import os

def get_apt_packages():
    if not os.path.exists("packages.txt"):
        return []
    with open("packages.txt", "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

apt_deps = get_apt_packages()

app = modal.App("my-ai-app")

image = (
    modal.Image.debian_slim()
    .apt_install(*apt_deps)
    .pip_install_from_requirements("requirements.txt")
    .add_local_python_source("app")
)

@app.function(image=image)
@modal.asgi_app()
def api():
    from app.main import app as fastapi_app
    return fastapi_app
