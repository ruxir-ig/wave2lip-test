from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import tempfile
import os
import uuid

from inference import load_models, make_animation, AnimationConfig

app = FastAPI(
    title="First-Order-Model Animation API",
    description="Animate a source image using a driving video",
    version="1.0.0"
)

# Global model state
generator = None
kp_detector = None

def reload_models():
    """Load or reload the models."""
    global generator, kp_detector
    generator, kp_detector = load_models()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    """Load models on startup."""
    reload_models()

@app.get("/")
def read_root():
    """Root endpoint with API info."""
    return {
        "message": "Welcome to the First-Order-Model Animation API",
        "endpoints": {
            "/health": "Health check",
            "/animate": "POST - Animate source image with driving video",
            "/reload_model": "Reload model weights"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": generator is not None and kp_detector is not None
    }

@app.get("/reload_model")
def reload_model_endpoint():
    """Reload model weights."""
    reload_models()
    return {"status": "models reloaded"}

@app.post("/animate")
async def animate(
    source_image: UploadFile = File(..., description="Source image (jpg, png)"),
    driving_video: UploadFile = File(..., description="Driving video (mp4)"),
    find_best_frame: bool = True,
    relative: bool = True,
    adapt_scale: bool = True
):
    """
    Animate a source image using a driving video.
    
    - **source_image**: The face image to animate
    - **driving_video**: Video providing the motion
    - **find_best_frame**: Find the best aligned frame (recommended for faces)
    - **relative**: Use relative keypoint movement
    - **adapt_scale**: Adapt movement scale based on face size
    
    Returns the animated video as MP4.
    """
    if generator is None or kp_detector is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    
    # Create temp directory for processing
    temp_dir = tempfile.mkdtemp()
    unique_id = str(uuid.uuid4())[:8]
    
    try:
        # Save uploaded files
        source_path = os.path.join(temp_dir, f"source_{unique_id}.png")
        video_path = os.path.join(temp_dir, f"driving_{unique_id}.mp4")
        output_path = os.path.join(temp_dir, f"output_{unique_id}.mp4")
        
        with open(source_path, "wb") as f:
            content = await source_image.read()
            f.write(content)
        
        with open(video_path, "wb") as f:
            content = await driving_video.read()
            f.write(content)
        
        # Run animation
        config = AnimationConfig(
            find_best_frame=find_best_frame,
            relative=relative,
            adapt_scale=adapt_scale
        )
        
        make_animation(
            source_path=source_path,
            driving_path=video_path,
            output_path=output_path,
            generator=generator,
            kp_detector=kp_detector,
            config=config
        )
        
        if not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="Animation generation failed")
        
        return FileResponse(
            output_path,
            media_type="video/mp4",
            filename="animated.mp4"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
