import torch
from src.models.pruning_model import VisionTransformerUCB

def test_token_merging():
    """
    Tests the token merging functionality in the VisionTransformerUCB model.
    """
    print("--- Testing Token Merging ---")

    # --- Model Configuration ---
    # Use a low keep_ratio to ensure pruning and merging occur.
    keep_ratio = 0.25
    n_classes = 10
    
    # Instantiate the model. pretrained=False to avoid dependency on the original model weights for this test.
    print(f"Initializing VisionTransformerUCB with keep_ratio={keep_ratio}...")
    model = VisionTransformerUCB(
        model_name="vit_tiny_patch16_224", # Using a standard small timm model for architecture
        pretrained=False, 
        n_classes=n_classes,
        keep_ratio=keep_ratio,
        selection_mode='ucb' # or 'random'
    )
    model.eval() # Set to evaluation mode

    # --- Monkey-patch the forward method of UCBEncoderBlock to print shapes ---
    original_forward = model.blocks[0].forward
    
def patched_forward(x, counter, pruning_enabled, ucb_count_score, selection_mode):
        print(f"\n--- Entering UCBEncoderBlock ---")
        print(f"Input shape: {x.shape}")
        
        # Call the original forward method to get the output
        x_final, score_delta = original_forward(x, counter, pruning_enabled, ucb_count_score, selection_mode)
        
        print(f"Output shape: {x_final.shape}")
        print(f"--- Exiting UCBEncoderBlock ---")
        
        # Check if sequence length was reduced as expected
        N_original = x.shape[1]
        N_final = x_final.shape[1]
        
        if pruning_enabled and counter > 50:
            # Expected new length: CLS + kept_patches + merged_token
            num_patches = N_original - 1 # Assuming exclude_cls=True
            expected_kept = max(1, int(num_patches * keep_ratio))
            
            # The 'removed_token_indices_list' is empty if all patches are kept.
            # So, if num_patches > expected_kept, there will be one merged token.
            expected_merged_tokens = 1 if num_patches > expected_kept else 0
            
            expected_N_final = 1 + expected_kept + expected_merged_tokens # CLS token + kept patches + (optional) merged token
            
            print(f"Expected final sequence length: {expected_N_final} (1 CLS + {expected_kept} kept + {expected_merged_tokens} merged)")
            assert N_final == expected_N_final, f"Shape mismatch! Expected {expected_N_final}, got {N_final}"
            print("Sequence length reduction is correct.")

        return x_final, score_delta

    # We only need to patch one block to verify the mechanism
    # Note: This patching will only affect the first block. 
    # For a full integration test across all blocks, the patching logic would need to be applied to all blocks.
    model.blocks[0].forward = patched_forward
    print("Patched the forward method of the first UCBEncoderBlock to print shapes.")

    # --- Create a dummy input ---
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    print(f"\nCreated a dummy input tensor of shape: {input_tensor.shape}")

    # --- Run a forward pass ---
    print("\nRunning a forward pass with pruning enabled...")
    with torch.no_grad():
        # Use a high counter to ensure pruning is active
        # The counter needs to be > 50 for pruning to be enabled in UCBAttention
        logits = model.forward_dynamic(pixel_values=input_tensor, counter=100, pruning_enabled=True)

    print(f"\nFinal logits shape: {logits.shape}")
    assert logits.shape == (batch_size, n_classes)

    print("\n--- Test Completed Successfully ---")
    print("The model ran without errors and the sequence length was reduced as expected in the patched block.")
    print("This indicates that the token merging logic is likely working correctly.")
    print("To run this test, save it as `scripts/test_token_merging.py` and execute `python -m scripts.test_token_merging` from the root directory.")

if __name__ == "__main__":
    test_token_merging()
