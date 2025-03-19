import os
import base64
import tempfile
import shutil
import uuid
import random
from typing import List, Optional, Union
from pathlib import Path
import logging
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np
from fastapi import FastAPI, Depends, HTTPException, status, File, UploadFile, Form, BackgroundTasks
from fastapi.security.api_key import APIKeyHeader, APIKey
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from PIL import Image
import imageio

# Import from inference.py instead of defining references to missing functions
from inference import (
    seed_everething, calculate_padding, load_image_to_tensor_with_resize_and_crop, 
    load_vae, load_unet, load_scheduler, convert_prompt_to_filename, get_unique_filename
)

# Import directly from packages
from ltx_video.pipelines.pipeline_ltx_video import LTXVideoPipeline
from ltx_video.utils.conditioning_method import ConditioningMethod
from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy
from ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier
from transformers import T5EncoderModel, T5Tokenizer
from q8_kernels.graph.graph import make_dynamic_graphed_callable

# Health check endpoint for dstack
class HealthCheck(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("api.log"), logging.StreamHandler()]
)
logger = logging.getLogger("ltx-video-api")

# API Key authorization
API_KEY = os.environ.get("LTX_VIDEO_API_KEY", "your-secret-key")  # Replace with secure key in production
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Setup temporary directory for uploaded files
TEMP_DIR = Path(os.environ.get("TEMP_UPLOADS_DIR", "/data/temp_uploads"))
TEMP_DIR.mkdir(exist_ok=True)

# Setup directory for output videos
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/data/outputs"))
OUTPUT_DIR.mkdir(exist_ok=True)

# Setup directory for static file serving
STATIC_DIR = Path(os.environ.get("STATIC_DIR", "/data/static"))
STATIC_DIR.mkdir(exist_ok=True)

# Q8 specific settings
LOW_VRAM = os.environ.get("LOW_VRAM", "true").lower() == "true"
TRANSFORMER_TYPE = os.environ.get("TRANSFORMER_TYPE", "q8_kernels")

# Model cache
loaded_model = None
model_lock = False

# Request models
class VideoGenerationRequest(BaseModel):
    prompt: str
    negative_prompt: str = "worst quality, inconsistent motion, blurry, jittery, distorted"
    seed: int = 171198
    num_inference_steps: int = 40
    num_frames: int = 121
    height: int = 480
    width: int = 704
    frame_rate: int = 25
    guidance_scale: float = 3.0
    low_vram: bool = LOW_VRAM
    transformer_type: str = TRANSFORMER_TYPE

class VideoGenerationResponse(BaseModel):
    job_id: str
    status: str = "processing"
    message: str

class VideoGenerationResult(BaseModel):
    job_id: str
    status: str
    video_url: Optional[str] = None
    error_message: Optional[str] = None

# Job tracking
active_jobs = {}

class JobStatus:
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

app = FastAPI(title="LTX-Video Q8 API", description="API for text-to-video generation with LTX-Video Q8")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key is None or api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key"
        )
    return api_key

def get_device():
    """Return the appropriate device for model operations."""
    return "cuda" if torch.cuda.is_available() else "cpu"

def save_upload_file_temporarily(upload_file: UploadFile) -> Path:
    """Save an upload file temporarily and return the path."""
    temp_file = Path(TEMP_DIR) / f"{uuid.uuid4()}_{upload_file.filename}"
    
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
        
    return temp_file

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint for dstack monitoring"""
    return HealthCheck()

@app.post("/api/generate", response_model=VideoGenerationResponse)
async def generate_video(
    background_tasks: BackgroundTasks,
    prompt: str = Form(...),
    negative_prompt: str = Form("worst quality, inconsistent motion, blurry, jittery, distorted"),
    seed: int = Form(171198),
    num_inference_steps: int = Form(40),
    num_frames: int = Form(121),
    height: int = Form(480),
    width: int = Form(704),
    frame_rate: int = Form(25),
    guidance_scale: float = Form(3.0),
    low_vram: bool = Form(LOW_VRAM),
    transformer_type: str = Form(TRANSFORMER_TYPE),
    conditioning_file: UploadFile = File(None),
    api_key: APIKey = Depends(get_api_key)
):
    job_id = str(uuid.uuid4())
    logger.info(f"Starting job {job_id} for prompt: {prompt}")
    
    # Save conditioning image if provided
    conditioning_image_path = None
    if conditioning_file and conditioning_file.filename:
        temp_file = await save_upload_file_temporarily(conditioning_file)
        conditioning_image_path = str(temp_file)
    
    # Store job parameters
    job_params = {
        "job_id": job_id,
        "ckpt_dir": os.environ.get("LTX_VIDEO_CKPT_PATH", "./models"),
        "unet_path": os.environ.get("LTX_VIDEO_UNET_PATH", "konakona/ltxvideo_q8"),
        "temp_file": conditioning_image_path,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "num_inference_steps": num_inference_steps,
        "num_frames": num_frames,
        "height": height,
        "width": width,
        "frame_rate": frame_rate,
        "guidance_scale": guidance_scale,
        "output_dir": str(OUTPUT_DIR),
        "device": get_device(),
        "low_vram": low_vram,
        "transformer_type": transformer_type
    }
    
    active_jobs[job_id] = {"status": JobStatus.PROCESSING, "result": None}
    
    # Run the generation in the background
    background_tasks.add_task(
        process_video_generation,
        job_params
    )
    
    return VideoGenerationResponse(
        job_id=job_id,
        status=JobStatus.PROCESSING,
        message="Video generation started"
    )

@app.get("/api/jobs/{job_id}", response_model=VideoGenerationResult)
async def check_job_status(job_id: str, api_key: APIKey = Depends(get_api_key)):
    if job_id not in active_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )
    
    job = active_jobs[job_id]
    
    return VideoGenerationResult(
        job_id=job_id,
        status=job["status"],
        video_url=job.get("result"),
        error_message=job.get("error")
    )

@app.get("/api/video/{video_name}")
async def get_video(video_name: str, api_key: APIKey = Depends(get_api_key)):
    video_path = Path(STATIC_DIR) / video_name
    
    if not video_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found"
        )
    
    return FileResponse(video_path)

@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str, api_key: APIKey = Depends(get_api_key)):
    if job_id not in active_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )
    
    # Delete any associated files
    if active_jobs[job_id].get("result"):
        video_path = Path(STATIC_DIR) / os.path.basename(active_jobs[job_id]["result"])
        if video_path.exists():
            try:
                os.remove(video_path)
            except Exception as e:
                logger.error(f"Error deleting video file: {e}")
    
    # Remove job from tracking
    del active_jobs[job_id]
    
    return {"message": f"Job {job_id} deleted"}

def process_video_generation(job_params):
    job_id = job_params["job_id"]
    temp_file = job_params.pop("temp_file", None)
    
    try:
        seed_everething(job_params["seed"])
        
        # Calculate padded dimensions
        height = job_params["height"]
        width = job_params["width"]
        num_frames = job_params["num_frames"]
        
        height_padded = ((height - 1) // 32 + 1) * 32
        width_padded = ((width - 1) // 32 + 1) * 32
        num_frames_padded = ((num_frames - 2) // 8 + 1) * 8 + 1
        
        padding = calculate_padding(height, width, height_padded, width_padded)
        
        # Load conditioning image if provided
        media_items = None
        if temp_file:
            media_items_prepad = load_image_to_tensor_with_resize_and_crop(
                temp_file, height, width
            )
            media_items = F.pad(
                media_items_prepad, padding, mode="constant", value=-1
            )
        
        # Paths for the separate mode directories
        ckpt_dir = Path(job_params["ckpt_dir"])
        unet_path = job_params["unet_path"]
        vae_dir = ckpt_dir / "vae"
        scheduler_dir = ckpt_dir / "scheduler"
        
        # Load models
        vae = load_vae(vae_dir)
        unet = load_unet(unet_path, type=job_params["transformer_type"])
        scheduler = load_scheduler(scheduler_dir)
        patchifier = SymmetricPatchifier(patch_size=1)
        text_encoder = T5EncoderModel.from_pretrained(
            ckpt_dir, subfolder="text_encoder", torch_dtype=torch.bfloat16
        )
        if torch.cuda.is_available() and not job_params["low_vram"]:
            text_encoder = text_encoder.to("cuda")
        
        tokenizer = T5Tokenizer.from_pretrained(
            "PixArt-alpha/PixArt-XL-2-1024-MS", subfolder="tokenizer"
        )
        
        unet = unet.to(torch.bfloat16)
        if job_params["transformer_type"] == "q8_kernels":
            for b in unet.transformer_blocks:
                b.to(dtype=torch.float)
            
            for n, m in unet.transformer_blocks.named_parameters():
                if "scale_shift_table" in n:
                    m.data = m.data.to(torch.bfloat16)
            
            torch.cuda.synchronize()
            unet.forward = make_dynamic_graphed_callable(unet.forward)
        
        # Use submodels for the pipeline
        submodel_dict = {
            "transformer": unet,
            "patchifier": patchifier,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "scheduler": scheduler,
            "vae": vae,
        }
        
        pipeline = LTXVideoPipeline(**submodel_dict)
        if torch.cuda.is_available() and not job_params["low_vram"]:
            pipeline = pipeline.to("cuda")
        
        # Prepare input for the pipeline
        sample = {
            "prompt": job_params["prompt"],
            "prompt_attention_mask": None,
            "negative_prompt": job_params["negative_prompt"],
            "negative_prompt_attention_mask": None,
            "media_items": media_items,
        }
        
        device = job_params["device"]
        generator = torch.Generator(device=device).manual_seed(job_params["seed"])
        
        # Generate the video
        images = pipeline(
            num_inference_steps=job_params["num_inference_steps"],
            num_images_per_prompt=1,
            guidance_scale=job_params["guidance_scale"],
            generator=generator,
            output_type="pt",
            callback_on_step_end=None,
            height=height_padded,
            width=width_padded,
            num_frames=num_frames_padded,
            frame_rate=job_params["frame_rate"],
            **sample,
            is_video=True,
            vae_per_channel_normalize=True,
            conditioning_method=(
                ConditioningMethod.FIRST_FRAME
                if media_items is not None
                else ConditioningMethod.UNCONDITIONAL
            ),
            mixed_precision=False,
            low_vram=job_params["low_vram"],
            transformer_type=job_params["transformer_type"]
        ).images
        
        # Crop the padded images to the desired resolution and number of frames
        (pad_left, pad_right, pad_top, pad_bottom) = padding
        pad_bottom = -pad_bottom
        pad_right = -pad_right
        if pad_bottom == 0:
            pad_bottom = images.shape[3]
        if pad_right == 0:
            pad_right = images.shape[4]
        images = images[:, :, :num_frames, pad_top:pad_bottom, pad_left:pad_right]
        
        # Process and save output
        video_np = images[0].permute(1, 2, 3, 0).cpu().float().numpy()
        # Unnormalizing images to [0, 255] range
        video_np = (video_np * 255).astype(np.uint8)
        fps = job_params["frame_rate"]
        
        # Get a unique filename for the output video
        if media_items is not None:
            base_filename = f"img_to_vid_{job_id}"
        else:
            base_filename = f"text_to_vid_{job_id}"
            
        output_filename = get_unique_filename(
            base_filename,
            ".mp4",
            prompt=job_params["prompt"],
            seed=job_params["seed"],
            resolution=(height, width, num_frames),
            dir=Path(job_params["output_dir"]),
        )
        
        # Write video
        with imageio.get_writer(output_filename, fps=fps) as video:
            for frame in video_np:
                video.append_data(frame)
        
        # Copy to static directory for serving
        static_filename = f"video_{job_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.mp4"
        static_path = Path(STATIC_DIR) / static_filename
        shutil.copy(output_filename, static_path)
        
        # Update job status
        active_jobs[job_id] = {
            "status": JobStatus.COMPLETED,
            "result": f"/static/{static_filename}"
        }
        
        logger.info(f"Job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Error processing job {job_id}: {str(e)}", exc_info=True)
        active_jobs[job_id] = {
            "status": JobStatus.FAILED,
            "error": str(e)
        }
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception as e:
                logger.error(f"Error removing temp file {temp_file}: {e}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    
    # Check if model path environment variable is set
    if not os.environ.get("LTX_VIDEO_CKPT_PATH"):
        logger.warning("LTX_VIDEO_CKPT_PATH environment variable not set. Using default ./models")
    
    # Verify API key is set and warn if using default
    if API_KEY == "your-secret-key":
        logger.warning("Using default API key. For production, set LTX_VIDEO_API_KEY environment variable")
    
    # Log Q8 specific settings
    logger.info(f"Starting LTX-Video Q8 API server with low_vram={LOW_VRAM}, transformer_type={TRANSFORMER_TYPE} on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)