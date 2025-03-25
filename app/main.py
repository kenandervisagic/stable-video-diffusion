import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model with CUDA/CPU-compatible settings
pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid", 
    torch_dtype=torch.float32
)
pipe.to(device)  # Move model to GPU if available

# Load the conditioning image
image = load_image("/app/input_image.png")
image = image.resize((1024, 576))

# Set a random seed for reproducibility
generator = torch.manual_seed(42)

# Generate video frames
frames = pipe(image, decode_chunk_size=4, generator=generator).frames[0]

# Export frames to a video file
export_to_video(frames, "generated.mp4", fps=7)

print("Video generated and saved as 'generated.mp4'")

