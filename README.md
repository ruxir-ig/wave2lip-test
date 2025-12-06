# First-Order-Model Face Animation API

FastAPI endpoint for animating a source image using a driving video, based on the First-Order Motion Model for Image Animation.

## Credits

This project is based on the work from:

- **Original Colab Notebook**: [fomm_bibi.ipynb](https://colab.research.google.com/github/eyaler/avatars4all/blob/master/fomm_bibi.ipynb)
- **First-Order-Model**: [eyaler/first-order-model](https://github.com/eyaler/first-order-model)

## API Endpoints

| Endpoint        | Method | Description                           |
| --------------- | ------ | ------------------------------------- |
| `/`             | GET    | API info and available endpoints      |
| `/health`       | GET    | Health check                          |
| `/animate`      | POST   | Generate animation from image + video |
| `/reload_model` | GET    | Reload model weights                  |

## Quick Start

### Using Docker (Recommended)

```bash
# Build the image
cd backend
docker build -t wave2lip-api .

# Run the container
docker run -p 8000:8000 wave2lip-api
```

### Manual Setup

```bash
cd backend
pip install -r requirements.txt

# Clone first-order-model if not present
git clone --depth 1 https://github.com/eyaler/first-order-model ../first-order-model

# Download weights
wget https://openavatarify.s3.amazonaws.com/weights/vox-adv-cpk.pth.tar \
    -O ../first-order-model/vox-adv-cpk.pth.tar

# Run the server
fastapi run main.py --port 8000
```

## Usage

### Animate Endpoint

```bash
curl -X POST http://localhost:8000/animate \
  -F "source_image=@face.jpg" \
  -F "driving_video=@video.mp4" \
  -o animated.mp4
```

### With Options

```bash
curl -X POST "http://localhost:8000/animate?find_best_frame=true&relative=true&adapt_scale=true" \
  -F "source_image=@face.jpg" \
  -F "driving_video=@video.mp4" \
  -o animated.mp4
```

### Health Check

```bash
curl http://localhost:8000/health
# {"status": "healthy", "models_loaded": true}
```

## API Parameters

| Parameter         | Type | Default  | Description                       |
| ----------------- | ---- | -------- | --------------------------------- |
| `source_image`    | File | required | Face image to animate (jpg, png)  |
| `driving_video`   | File | required | Video providing motion (mp4)      |
| `find_best_frame` | bool | true     | Find best aligned frame           |
| `relative`        | bool | true     | Use relative keypoint movement    |
| `adapt_scale`     | bool | true     | Adapt movement scale to face size |

## Lightning.ai Deployment

1. Push this repository to GitHub
2. Create a new Lightning.ai Studio
3. Connect your GitHub repository
4. Build using the Dockerfile in `backend/`
5. Expose port 8000
