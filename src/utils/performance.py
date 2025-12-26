import torch
import time
from functools import partial

# Add a check for fvcore and print a helpful message if it's not installed.
try:
    from fvcore.nn import FlopCountAnalysis
except ImportError:
    print("fvcore is not installed. To calculate FLOPS, please install it via: pip install fvcore")
    FlopCountAnalysis = None

def calculate_performance_metrics(model, input_size=(3, 224, 224), device='cuda', keep_ratio=None):
    """
    Calculates FLOPS and inference time for a given model.
    The model is configured for pruned inference (merging logic) for the calculation.

    Args:
        model: The PyTorch model, expected to be VisionTransformerUCB.
        input_size: A tuple representing the input size (C, H, W).
        device: The device to run the model on ('cuda' or 'cpu').
        keep_ratio: The keep_ratio the model is configured with. Used for display.

    Returns:
        A dictionary containing GFLOPS and inference time in ms.
    """
    if not torch.cuda.is_available() and device == 'cuda':
        print("CUDA not available, switching to CPU for performance metrics.")
        device = 'cpu'
        
    model.to(device)
    model.eval()

    # Create a forward function for performance analysis that ensures pruning is active.
    # We need a high counter value to exit the warm-up period.
    # This partial function will be called by FlopCountAnalysis and the timing loop.
    analysis_forward = partial(model.forward_dynamic, counter=100, pruning_enabled=True, labels=None)

    dummy_input = torch.randn(1, *input_size).to(device)

    # --- Calculate FLOPS ---
    gflops = 0
    if FlopCountAnalysis:
        try:
            # fvcore's FlopCountAnalysis needs an nn.Module. We wrap our partial function call.
            class ForwardWrapper(torch.nn.Module):
                def __init__(self, fn):
                    super().__init__()
                    self.fn = fn
                def forward(self, x):
                    return self.fn(pixel_values=x)
            
            wrapped_model = ForwardWrapper(analysis_forward)
            flops = FlopCountAnalysis(wrapped_model, dummy_input)
            gflops = flops.total() / 1e9
            print(f"GFLOPS for keep_ratio={keep_ratio}: {gflops:.2f}")

        except Exception as e:
            print(f"Could not calculate FLOPS: {e}")
            gflops = 0
    else:
        print("Skipping FLOPS calculation because fvcore is not installed.")


    # --- Calculate Inference Time ---
    # Warm-up
    for _ in range(10):
        with torch.no_grad():
            analysis_forward(pixel_values=dummy_input)

    # Measurement
    num_runs = 50
    if device == 'cuda':
        torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(num_runs):
        with torch.no_grad():
            analysis_forward(pixel_values=dummy_input)

    if device == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()

    avg_inference_time_ms = ((end_time - start_time) / num_runs) * 1000
    print(f"Inference time for keep_ratio={keep_ratio}: {avg_inference_time_ms:.2f} ms")


    return {
        "gflops": gflops,
        "inference_ms": avg_inference_time_ms
    }