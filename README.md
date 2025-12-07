First Order Motion Model – FastAPI deployment helper
===================================================

This repo adapts the First Order Motion Model demo into a FastAPI service for animating a source face image using a driving video. See `api/` for the service code and `book.ipynb` for the original exploratory notebook.

Credits
-------
- Original notebook and workflow inspiration: https://colab.research.google.com/github/eyaler/avatars4all/blob/master/fomm_bibi.ipynb
- Model code: https://github.com/AliaksandrSiarohin/first-order-model

Getting Started (local)
-----------------------
1) Create a virtualenv and install requirements:
	```bash
	python -m venv venv
	source venv/bin/activate
	pip install -r api/requirements.txt
	```
2) Download weights into `first-order-model/` (e.g., `vox-adv-cpk.pth.tar`).
3) Run the API:
	```bash
	cd api
	uvicorn app:app --host 0.0.0.0 --port 8000
	```
4) Open http://localhost:8000/docs to test.

Docker (GPU)
------------
```bash
docker build -t fomm-api .
docker run --gpus all -p 8000:8000 \
  -v $(pwd)/first-order-model/vox-adv-cpk.pth.tar:/app/first-order-model/vox-adv-cpk.pth.tar \
  fomm-api
```

Deployment on Lightning.ai
--------------------------

### Lightning Studios Setup

Add the following to your `on_start.sh` script in Lightning Studios (located at `~/.lightning_studio/on_start.sh`):

```bash
# Start Docker service
service docker start

# Navigate to project
cd ~/wave2lip-test/backend

# Build and run Docker container
docker build -t wave2lip-api .
docker run -d -p 8000:8000 wave2lip-api

# Wait for API to be ready
sleep 10

# Start Gradio interface
cd ~/wave2lip-test
python gradio_app.py
```

This will automatically start your API and Gradio interface every time the Studio starts.

Security Notes
--------------
- Validate input sizes and types; keep uploads small.
- Run containers as non-root when possible.
- Clean temporary files and avoid bundling large checkpoints inside images; mount them instead.
