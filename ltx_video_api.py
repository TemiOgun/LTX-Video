import os
import base64
import tempfile
import shutil
import uuid
from typing import List, Optional, Union
from pathlib import Path
import logging
from datetime import datetime

import torch
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

# Import the necessary functions from your inference.py
from ltx_video.pipelines.pipeline_ltx_video import ConditioningItem, LTXVideoPipeline
from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy

# Import these functions from your inference.py
from inference import (
    get_device, seed_everething, calculate_padding, 
    load_image_to_tensor_with_resize_and_crop, 
    create_ltx_video_pipeline, prepare_conditioning, 
    get_unique_filename
)

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
    stg_scale: float = 1.0
    stg_rescale: float = 0.7
    stg_mode: str = "attention_values"
    stg_skip_layers: str = "19"
    image_cond_noise_scale: float = 0.15
    decode_timestep: float = 0.025
    decode_noise_scale: float = 0.0125
    precision: str = "bfloat16"
    sampler: Optional[str] = None
    prompt_enhancement_words_threshold: int = 50
    low_vram: bool = LOW_VRAM
    transformer_type: str = TRANSFORMER_TYPE

class ConditioningMedia(BaseModel):
    start_frame: int
    strength: float = 1.0
    
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
    stg_scale: float = Form(1.0),
    stg_rescale: float = Form(0.7),
    stg_mode: str = Form("attention_values"),
    stg_skip_layers: str = Form("19"),
    image_cond_noise_scale: float = Form(0.15),
    decode_timestep: float = Form(0.025),
    decode_noise_scale: float = Form(0.0125),
    precision: str = Form("bfloat16"),
    low_vram: bool = Form(LOW_VRAM),
    transformer_type: str = Form(TRANSFORMER_TYPE),
    conditioning_files: List[UploadFile] = File(None),
    conditioning_strengths: str = Form(None),  # Comma-separated list of floats
    conditioning_start_frames: str = Form(None),  # Comma-separated list of integers
    api_key: APIKey = Depends(get_api_key)
):
    job_id = str(uuid.uuid4())
    logger.info(f"Starting job {job_id} for prompt: {prompt}")
    
    # Parse lists from form data
    temp_files = []
    conditioning_media_paths = []
    
    if conditioning_files:
        for upload_file in conditioning_files:
            if upload_file.filename:  # Skip if no file provided
                temp_file = await save_upload_file_temporarily(upload_file)
                temp_files.append(temp_file)
                conditioning_media_paths.append(str(temp_file))
    
    parsed_strengths = []
    if conditioning_strengths:
        parsed_strengths = [float(s.strip()) for s in conditioning_strengths.split(",")]
        
    parsed_start_frames = []
    if conditioning_start_frames:
        parsed_start_frames = [int(s.strip()) for s in conditioning_start_frames.split(",")]
    
    # Validate conditioning parameters
    if conditioning_media_paths:
        if len(parsed_strengths) != len(conditioning_media_paths):
            parsed_strengths = [1.0] * len(conditioning_media_paths)
        if len(parsed_start_frames) != len(conditioning_media_paths):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Number of conditioning start frames must match number of conditioning files"
            )
    
    # Store job parameters
    job_params = {
        "job_id": job_id,
        "ckpt_path": os.environ.get("LTX_VIDEO_CKPT_PATH", "./models/ltx_video_q8_model.safetensors"),
        "text_encoder_model_name_or_path": os.environ.get("LTX_VIDEO_TEXT_ENCODER", "PixArt-alpha/PixArt-XL-2-1024-MS"),
        "temp_files": temp_files,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "num_inference_steps": num_inference_steps,
        "num_frames": num_frames,
        "height": height,
        "width": width,
        "frame_rate": frame_rate,
        "guidance_scale": guidance_scale,
        "stg_scale": stg_scale,
        "stg_rescale": stg_rescale,
        "stg_mode": stg_mode,
        "stg_skip_layers": stg_skip_layers,
        "image_cond_noise_scale": image_cond_noise_scale,
        "decode_timestep": decode_timestep,
        "decode_noise_scale": decode_noise_scale,
        "precision": precision,
        "conditioning_media_paths": conditioning_media_paths,
        "conditioning_strengths": parsed_strengths,
        "conditioning_start_frames": parsed_start_frames,
        "output_path": str(OUTPUT_DIR),
        "offload_to_cpu": True,
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
    temp_files = job_params.pop("temp_files", [])
    
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
        
        # Create pipeline
        prompt_word_count = len(job_params["prompt"].split())
        prompt_enhancement_threshold = job_params.get("prompt_enhancement_words_threshold", 50)
        enhance_prompt = (
            prompt_enhancement_threshold > 0
            and prompt_word_count < prompt_enhancement_threshold
        )

        global loaded_model, model_lock
        
        # Load model if not already loaded (simple caching)
        if loaded_model is None and not model_lock:
            model_lock = True
            try:
                # Set up model paths for the Q8 version
                base_model_dir = os.path.dirname(job_params["ckpt_path"])
                model_paths = {
                    "transformer": job_params["ckpt_path"],
                    "vae": os.path.join(base_model_dir, "vae"),
                    "text_encoder": os.path.join(base_model_dir, "text_encoder"),
                    "scheduler": os.path.join(base_model_dir, "scheduler"),
                }
                
                logger.info(f"Using model paths: {model_paths}")
                
                # Add Q8-specific parameters to the pipeline creation
                loaded_model = create_ltx_video_pipeline(
                    ckpt_path=job_params["ckpt_path"],
                    precision=job_params["precision"],
                    text_encoder_model_name_or_path=job_params["text_encoder_model_name_or_path"],
                    sampler=job_params.get("sampler"),
                    device=job_params["device"],
                    enhance_prompt=enhance_prompt,
                    low_vram=job_params.get("low_vram", True),
                    transformer_type=job_params.get("transformer_type", "q8_kernels"),
                    # Provide explicit paths to each model component
                    vae_path=model_paths["vae"],
                    text_encoder_path=model_paths["text_encoder"],
                    scheduler_path=model_paths["scheduler"]
                )
            finally:
                model_lock = False
                
        pipeline = loaded_model
        
        # Prepare conditioning items
        conditioning_items = None
        if job_params.get("conditioning_media_paths"):
            conditioning_items = prepare_conditioning(
                conditioning_media_paths=job_params["conditioning_media_paths"],
                conditioning_strengths=job_params["conditioning_strengths"],
                conditioning_start_frames=job_params["conditioning_start_frames"],
                height=height,
                width=width,
                num_frames=num_frames,
                padding=padding,
                pipeline=pipeline
            )
        
        # Set spatiotemporal guidance
        skip_block_list = [int(x.strip()) for x in job_params["stg_skip_layers"].split(",")]
        stg_mode = job_params["stg_mode"].lower()
        
        if stg_mode == "attention_values" or stg_mode == "stg_av":
            skip_layer_strategy = SkipLayerStrategy.AttentionValues
        elif stg_mode == "attention_skip" or stg_mode == "stg_as":
            skip_layer_strategy = SkipLayerStrategy.AttentionSkip
        elif stg_mode == "residual" or stg_mode == "stg_r":
            skip_layer_strategy = SkipLayerStrategy.Residual
        elif stg_mode == "transformer_block" or stg_mode == "stg_t":
            skip_layer_strategy = SkipLayerStrategy.TransformerBlock
        else:
            skip_layer_strategy = SkipLayerStrategy.AttentionValues
        
        # Prepare input for the pipeline
        sample = {
            "prompt": job_params["prompt"],
            "prompt_attention_mask": None,
            "negative_prompt": job_params["negative_prompt"],
            "negative_prompt_attention_mask": None,
        }

        device = job_params["device"]
        generator = torch.Generator(device=device).manual_seed(job_params["seed"])
        
        # Generate the video
        images = pipeline(
            num_inference_steps=job_params["num_inference_steps"],
            num_images_per_prompt=1,
            guidance_scale=job_params["guidance_scale"],
            skip_layer_strategy=skip_layer_strategy,
            skip_block_list=skip_block_list,
            stg_scale=job_params["stg_scale"],
            do_rescaling=job_params["stg_rescale"] != 1,
            rescaling_scale=job_params["stg_rescale"],
            generator=generator,
            output_type="pt",
            callback_on_step_end=None,
            height=height_padded,
            width=width_padded,
            num_frames=num_frames_padded,
            frame_rate=job_params["frame_rate"],
            **sample,
            conditioning_items=conditioning_items,
            is_video=True,
            vae_per_channel_normalize=True,
            image_cond_noise_scale=job_params["image_cond_noise_scale"],
            decode_timestep=job_params["decode_timestep"],
            decode_noise_scale=job_params["decode_noise_scale"],
            mixed_precision=(job_params["precision"] == "mixed_precision"),
            offload_to_cpu=job_params.get("offload_to_cpu", False),
            device=device,
            enhance_prompt=enhance_prompt,
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
        output_filename = get_unique_filename(
            f"video_{job_id}",
            ".mp4",
            prompt=job_params["prompt"],
            seed=job_params["seed"],
            resolution=(height, width, num_frames),
            dir=OUTPUT_DIR,
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
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                logger.error(f"Error removing temp file {temp_file}: {e}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    
    # Check if model path environment variable is set
    if not os.environ.get("LTX_VIDEO_CKPT_PATH"):
        logger.warning("LTX_VIDEO_CKPT_PATH environment variable not set. Using default ./models/ltx_video_q8_model.safetensors")
    
    # Verify API key is set and warn if using default
    if API_KEY == "your-secret-key":
        logger.warning("Using default API key. For production, set LTX_VIDEO_API_KEY environment variable")
    
    # Log Q8 specific settings
    logger.info(f"Starting LTX-Video Q8 API server with low_vram={LOW_VRAM}, transformer_type={TRANSFORMER_TYPE} on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)