import torch
from src.models.pruning_model import VisionTransformerUCB

def test_token_merging_with_ucb_update_control():
    """
    Tests the token merging functionality in the VisionTransformerUCB model
    with control over UCB score updates.
    """
    print("--- Testing Token Merging with UCB Update Control ---")

    # --- Model Configuration ---
    keep_ratio = 0.25 # Use a low keep_ratio to ensure pruning and merging occur. 
    n_classes = 10
    batch_size = 2
    
    # Instantiate the model. pretrained=False to avoid dependency on the original model weights for this test.
    print(f"Initializing VisionTransformerUCB with keep_ratio={keep_ratio}...")
    model = VisionTransformerUCB(
        model_name="vit_tiny_patch16_224", # Using a standard small timm model for architecture
        pretrained=False, 
        n_classes=n_classes,
        keep_ratio=keep_ratio,
        selection_mode='ucb'
    )
    model.eval() # Set to evaluation mode

    # Save initial ucb_count_scores for comparison
    initial_ucb_scores = model.ucb_count_scores.clone()
    print("Initial UCB count scores captured.")

    # --- Monkey-patch the forward method of UCBEncoderBlock to print shapes ---
    original_ucb_encoder_block_forward = model.blocks[0].forward
    
    def patched_forward(x, counter, pruning_enabled, ucb_count_score, selection_mode, ucb_update_enabled):
        print(f"\n--- Entering UCBEncoderBlock (ucb_update_enabled={ucb_update_enabled}) ---")
        print(f"Input shape: {x.shape}")
        
        x_final, score_delta = original_ucb_encoder_block_forward(x, counter, pruning_enabled, ucb_count_score, selection_mode, ucb_update_enabled)
        
        print(f"Output shape: {x_final.shape}")
        
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
        
        print(f"--- Exiting UCBEncoderBlock (ucb_update_enabled={ucb_update_enabled}) ---")
        
        return x_final, score_delta

    model.blocks[0].forward = patched_forward
    print("Patched the forward method of the first UCBEncoderBlock to print shapes.")

    # --- Create a dummy input ---
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    print(f"\nCreated a dummy input tensor of shape: {input_tensor.shape}")

    # --- Test Case 1: UCB scores are updated ---
    print("\n--- Running Test Case 1: UCB scores SHOULD be updated ---")
    current_ucb_scores_before_run = model.ucb_count_scores.clone()
    with torch.no_grad():
        _ = model(pixel_values=input_tensor, counter=100, pruning_enabled=True, ucb_update_enabled=True)
    
    ucb_scores_after_run1 = model.ucb_count_scores.clone()
    assert not torch.equal(current_ucb_scores_before_run, ucb_scores_after_run1), "UCB scores were NOT updated as expected in Test Case 1!"
    print("Test Case 1 passed: UCB scores were updated.")

    # --- Test Case 2: UCB scores are NOT updated ---
    print("\n--- Running Test Case 2: UCB scores SHOULD NOT be updated ---")
    current_ucb_scores_before_run2 = model.ucb_count_scores.clone()
    with torch.no_grad():
        _ = model(pixel_values=input_tensor, counter=100, pruning_enabled=True, ucb_update_enabled=False)
    
    ucb_scores_after_run2 = model.ucb_count_scores.clone()
    assert torch.equal(current_ucb_scores_before_run2, ucb_scores_after_run2), "UCB scores WERE updated unexpectedly in Test Case 2!"
    print("Test Case 2 passed: UCB scores were NOT updated.")


    print("\n--- All Tests Completed Successfully ---")
    print("The model ran without errors, sequence length reduction is correct, and UCB score update control works.")
    print("To run this test, save it as `scripts/test_token_merging.py` and execute `python -m scripts.test_token_merging` from the root directory.")

if __name__ == "__main__":
    test_token_merging_with_ucb_update_control()
