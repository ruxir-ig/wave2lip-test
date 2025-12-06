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
- Configure `lightning.yaml`, then deploy with the Lightning CLI or run inside a Lightning Studio GPU workspace.

Security Notes
--------------
- Validate input sizes and types; keep uploads small.
- Run containers as non-root when possible.
- Clean temporary files and avoid bundling large checkpoints inside images; mount them instead.
