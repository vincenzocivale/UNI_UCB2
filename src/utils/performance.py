import torch
import time
from fvcore.nn import FlopCountAnalysis, flop_count_table

def calculate_pruning_performance(model, dummy_input, top_k_indices, device, num_warmup=10, num_benchmark=50):
    """
    Calculates inference time and FLOPs for a pruned model.

    Args:
        model (torch.nn.Module): The pruned model.
        dummy_input (torch.Tensor): A dummy input tensor for FLOPs and inference time measurement.
        top_k_indices (torch.Tensor): The indices of patches to keep.
        device (torch.device): The device to run the measurements on.
        num_warmup (int): Number of warmup runs for inference time measurement.
        num_benchmark (int): Number of benchmark runs for inference time measurement.

    Returns:
        dict: A dictionary containing 'inference_time_ms' and 'flops'.
    """
    model.eval()
    model.to(device)
    dummy_input = dummy_input.to(device)
    top_k_indices = top_k_indices.to(device)

    # --- Measure Inference Time ---
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Warmup runs
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model.forward_pruned(dummy_input, top_k_indices)
        if device.type == 'cuda':
            torch.cuda.synchronize()

    # Benchmark runs
    total_time = 0
    with torch.no_grad():
        for _ in range(num_benchmark):
            if device.type == 'cuda':
                start_event.record()
            _ = model.forward_pruned(dummy_input, top_k_indices)
            if device.type == 'cuda':
                end_event.record()
                torch.cuda.synchronize()
                total_time += start_event.elapsed_time(end_event) # milliseconds
            else:
                start_time = time.perf_counter()
                _ = model.forward_pruned(dummy_input, top_k_indices)
                end_time = time.perf_counter()
                total_time += (end_time - start_time) * 1000 # convert to milliseconds

    avg_inference_time_ms = total_time / num_benchmark

    # --- Calculate FLOPs ---
    # Need to create a new model instance for FLOPs analysis to avoid interference with existing model state
    # or ensure the forward_pruned method is directly traceable.
    # For fvcore, we want to trace the actual computation path.
    # Let's assume model.forward_pruned is directly traceable.
    try:
        inputs_for_flops = (dummy_input, top_k_indices) # forward_pruned takes these two
        flops = FlopCountAnalysis(model, inputs_for_flops)
        total_flops = flops.total()
        # Optionally print a table for detailed FLOPs breakdown
        # print(flop_count_table(flops))
    except Exception as e:
        print(f"Warning: Could not calculate FLOPs: {e}")
        total_flops = -1 # Indicate failure

    return {
        'inference_time_ms': avg_inference_time_ms,
        'flops': total_flops
    }
