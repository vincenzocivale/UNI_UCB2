import torch
import torch.nn as nn
import timm

class UCBAttention(nn.Module):
    """
    Corrected UCB Attention module with proper patch pruning.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 k=None, keep_ratio=None, beta=1.0, exclude_cls=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.k = k
        self.keep_ratio = keep_ratio
        self.beta = beta
        self.exclude_cls = exclude_cls

    def ucb_score_pruning(self, attn_scores, attn_probs, v, iteration, count_score_buffer):
        """
        Applies UCB-based patch pruning on attention probabilities.
        Fixed to avoid NaN issues while maintaining correct behavior.
        """
        B, H, N, _ = attn_scores.shape
        device = v.device

        # Determine dynamic k (number of patches to keep)
        if self.exclude_cls:
            total_patches = N - 1
            patch_start_idx = 1
        else:
            total_patches = N
            patch_start_idx = 0

        if self.keep_ratio is not None:
            # FIX: Gestisci correttamente keep_ratio=0
            if self.keep_ratio <= 0:
                k = 0  # Prune TUTTE le patch (mantieni solo CLS)
            else:
                k = max(1, int(total_patches * self.keep_ratio))
        elif self.k is not None:
            k = min(self.k, total_patches)
        else:
            # If no pruning specified, return original attention
            return torch.matmul(self.attn_drop(attn_probs), v), None

        # Se k=0, restituisci solo l'attenzione sul CLS token
        if k == 0 and self.exclude_cls:
            # Crea una maschera che mantiene solo il CLS token
            mask = torch.zeros(B, H, N, N, device=device, dtype=attn_probs.dtype)
            mask[:, :, :, 0] = 1.0  # Tutti possono attendere al CLS
            mask[:, :, 0, :] = 1.0  # CLS può attendere a tutti
            
            pruned_attn = attn_probs * mask
            pruned_attn = pruned_attn / (pruned_attn.sum(dim=-1, keepdim=True) + 1e-8)
            pruned_attn = self.attn_drop(pruned_attn)
            context = torch.matmul(pruned_attn, v)
            
            # FIX: Nessun aggiornamento degli score perché nessuna patch è selezionata
            # Ma incrementa comunque il CLS token per tracking
            score_delta = torch.zeros_like(count_score_buffer, dtype=torch.float32)
            score_delta[:, 0] += 1.0  # Solo CLS viene "selezionato"
            return context, score_delta

        # Calculate patch importance scores - using attention received by each patch
        if self.exclude_cls:
            # Average attention each patch receives (excluding CLS)
            patch_scores = attn_probs[:, :, :, 1:].mean(dim=2)  # (B, H, N-1)
            relevant_counts = count_score_buffer[:, 1:]  # (H, N-1)
        else:
            patch_scores = attn_probs.mean(dim=2)  # (B, H, N)
            relevant_counts = count_score_buffer  # (H, N)

        # UCB exploration term
        ucb_exploration = self.beta * torch.sqrt(
            torch.log(torch.tensor(iteration + 1.0, device=device)) / (relevant_counts + 1e-6)
        )

        # Combine scores with exploration bonus
        ucb_scores = patch_scores + ucb_exploration.unsqueeze(0)  # (B, H, num_patches)

        # Global selection (same patches for all heads in each sample)
        global_ucb_scores = ucb_scores.mean(dim=1)  # (B, num_patches)
        _, selected_indices = torch.topk(global_ucb_scores, k=k, dim=-1)  # (B, k)

        # Create attention mask
        mask = torch.zeros(B, H, N, N, device=device, dtype=attn_probs.dtype)
        
        if self.exclude_cls:
            # Always preserve CLS token interactions
            mask[:, :, :, 0] = 1.0  # All tokens can attend to CLS
            mask[:, :, 0, :] = 1.0  # CLS can attend to all tokens
            
            # Convert patch indices to token indices (add 1 for CLS offset)
            token_indices = selected_indices + 1  # (B, k)
            
            for b in range(B):
                mask[b, :, :, token_indices[b]] = 1.0  # All can attend TO selected
                mask[b, :, token_indices[b], :] = 1.0  # Selected can attend to ALL
        else:
            for b in range(B):
                mask[b, :, :, selected_indices[b]] = 1.0  # All can attend TO selected
                mask[b, :, selected_indices[b], :] = 1.0  # Selected can attend to ALL

        pruned_attn = attn_probs * mask
        
        # Renormalize with epsilon to avoid division by zero
        pruned_attn = pruned_attn / (pruned_attn.sum(dim=-1, keepdim=True) + 1e-8)
        
        pruned_attn = self.attn_drop(pruned_attn)

        context = torch.matmul(pruned_attn, v)
        
        # Update counts for selected patches
        # FIX: Assicurati che score_delta sia un tensor float, non int
        score_delta = torch.zeros_like(count_score_buffer, dtype=torch.float32)
        
        if self.exclude_cls:
            for b in range(B):
                # Each selected patch gets +1/B to normalize across batch
                # FIX: Converti selected_indices in long per indexing sicuro
                indices_to_update = (selected_indices[b] + 1).long()
                score_delta[:, indices_to_update] += 1.0 / B
        else:
            for b in range(B):
                indices_to_update = selected_indices[b].long()
                score_delta[:, indices_to_update] += 1.0 / B

        return context, score_delta
    
    
    def forward(self, x, counter, ucb_enabled, ucb_count_score):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn_scores = (q @ k.transpose(-1, -2)) * self.scale
        attn_probs = attn_scores.softmax(dim=-1)

        score_delta = None
        # FIX: RIDOTTO WARMUP da 50 a 10 per attivare UCB prima
        if ucb_enabled and counter > 10:  # ← ERA 50, ORA 10
            context, score_delta = self.ucb_score_pruning(
                attn_scores, attn_probs, v, iteration=counter, count_score_buffer=ucb_count_score
            )
        else:
            # Nessun pruning - attention standard
            attn_probs = self.attn_drop(attn_probs)
            context = torch.matmul(attn_probs, v)

        x = context.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, score_delta


class UCBBlock(nn.Module):
    def __init__(self, original_block: nn.Module, **ucb_kwargs):
        super().__init__()
        # Copy all components from original block
        self.norm1 = original_block.norm1
        self.norm2 = original_block.norm2
        self.mlp = original_block.mlp
        
        # Handle different timm versions
        self.ls1 = getattr(original_block, 'ls1', nn.Identity())
        self.ls2 = getattr(original_block, 'ls2', nn.Identity())
        self.drop_path1 = getattr(original_block, 'drop_path1', nn.Identity())
        self.drop_path2 = getattr(original_block, 'drop_path2', nn.Identity())

        # Replace attention with UCB version
        orig_attn = original_block.attn
        self.attn = UCBAttention(
            dim=orig_attn.qkv.in_features,
            num_heads=orig_attn.num_heads,
            qkv_bias=orig_attn.qkv.bias is not None,
            attn_drop=orig_attn.attn_drop.p if hasattr(orig_attn.attn_drop, 'p') else 0.0,
            proj_drop=orig_attn.proj_drop.p if hasattr(orig_attn.proj_drop, 'p') else 0.0,
            **ucb_kwargs
        )
        
        # Load weights from original attention
        state_dict = orig_attn.state_dict()
        self.attn.load_state_dict(state_dict, strict=False)

    def forward(self, x, counter, ucb_enabled, ucb_count_score):
        # Standard transformer block with residual connections
        h, score_delta = self.attn(self.norm1(x), counter, ucb_enabled, ucb_count_score)
        x = x + self.drop_path1(self.ls1(h))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x, score_delta


class ViT_UCB_Pruning(nn.Module):
    def __init__(self, model_name="hf-hub:MahmoodLab/uni", pretrained=True, n_classes=None,
                 k=None, keep_ratio=None, beta=1.0, exclude_cls=True):
        super().__init__()
        print(f"Loading source model '{model_name}'...")
        source_model = timm.create_model(model_name, pretrained=pretrained, init_values=1e-5)

        self.keep_ratio = keep_ratio

        # Copy patch embedding and positional components
        self.patch_embed = source_model.patch_embed
        self.cls_token = source_model.cls_token
        self.pos_embed = source_model.pos_embed
        self.pos_drop = source_model.pos_drop
        self.norm = source_model.norm
        
        # Setup classification head
        if hasattr(source_model, 'head'):
            if hasattr(source_model.head, 'in_features'):
                head_dim = source_model.head.in_features
            else:
                head_dim = source_model.head.weight.shape[1] if hasattr(source_model.head, 'weight') else 1024
        else:
            head_dim = source_model.embed_dim if hasattr(source_model, 'embed_dim') else 1024
            
        self.head = nn.Linear(head_dim, n_classes if n_classes is not None else head_dim)
        self.n_classes = n_classes if n_classes is not None else head_dim

        # Replace blocks with UCB versions
        self.blocks = nn.ModuleList([
            UCBBlock(block, k=k, keep_ratio=keep_ratio, beta=beta, exclude_cls=exclude_cls) 
            for block in source_model.blocks
        ])

        # Initialize UCB count buffer
        num_layers = len(self.blocks)
        num_heads = self.blocks[0].attn.num_heads
        num_patches = self.pos_embed.shape[1]  # Total tokens including CLS

        # Buffer to track patch selection counts per layer and head
        self.register_buffer("ucb_count_scores", torch.ones(num_layers, num_heads, num_patches))

    def forward(self, x: torch.Tensor, counter: int = 0, ucb_enabled: bool = True, labels: torch.Tensor = None):
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        # DEBUG: Aggiungi logging SEMPRE per verificare counter e ucb_enabled
        if counter % 10 == 0:  # FIX: Log più frequente
            print(f"[DEBUG FORWARD] Counter={counter}, UCB_enabled={ucb_enabled}, Warmup_passed={counter > 50}, keep_ratio={self.keep_ratio}")

        # FIX: Rimuovi il pruning fisico preliminare, lascia che UCB lo gestisca nei blocchi
        # Process through blocks con UCB ABILITATO
        for i, block in enumerate(self.blocks):
            x, score_delta = block(
                x, 
                counter=counter, 
                ucb_enabled=ucb_enabled,  # FIX: Passa il vero valore di ucb_enabled
                ucb_count_score=self.ucb_count_scores[i]
            )
            
            # DEBUG: Verifica se score_delta viene calcolato
            if counter % 10 == 0 and i == 0:  # FIX: Log più frequente
                print(f"[DEBUG LAYER {i}] score_delta={'None' if score_delta is None else 'SET'}, ucb_enabled={ucb_enabled}, counter={counter}")
                if score_delta is not None:
                    print(f"[DEBUG LAYER {i}] score_delta shape: {score_delta.shape}, mean: {score_delta.mean().item():.6f}, max: {score_delta.max().item():.6f}")
                    print(f"[DEBUG LAYER {i}] UCB scores BEFORE - mean: {self.ucb_count_scores[i].mean().item():.6f}, std: {self.ucb_count_scores[i].std().item():.6f}")
                else:
                    print(f"[DEBUG LAYER {i}] NO PRUNING - counter={counter}, warmup_threshold=50")
            
            if score_delta is not None:
                # FIX: Usa .data per aggiornare il buffer senza interferire con autograd
                old_mean = self.ucb_count_scores[i].mean().item()
                self.ucb_count_scores[i].data += score_delta.data
                new_mean = self.ucb_count_scores[i].mean().item()
                
                if counter % 10 == 0 and i == 0:  # FIX: Log più frequente
                    print(f"[DEBUG LAYER {i}] UCB scores AFTER - mean: {new_mean:.6f} (delta: {new_mean - old_mean:.6f})")

        x = self.norm(x)
        logits = self.head(x[:, 0])

        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return loss, logits
        return logits

    def get_top_k_patch_indices(self, keep_ratio: float):
        """
        Analyzes the learned ucb_count_scores to find the most important patch indices.

        Args:
            keep_ratio (float): The ratio of patches to keep (e.g., 0.7 for 70%).

        Returns:
            torch.Tensor: A 1D tensor containing the indices of tokens to keep,
                          including the CLS token.
        """
        # Exclude CLS token from importance calculation
        # Scores are averaged across all layers and heads
        patch_scores = self.ucb_count_scores[:, :, 1:].mean(dim=(0, 1))
        
        # FIX: Gestisci keep_ratio=0
        if keep_ratio <= 0:
            # Ritorna solo il CLS token
            return torch.tensor([0], device=patch_scores.device)
        
        num_patches_to_keep = max(1, int(patch_scores.shape[0] * keep_ratio))
        
        # Get the indices of the patches with the highest scores
        top_patch_indices = torch.topk(patch_scores, k=num_patches_to_keep, dim=-1).indices
        
        # Add 1 to offset for the CLS token we excluded
        token_indices = top_patch_indices + 1
        
        # Always include the CLS token (index 0)
        cls_token_index = torch.tensor([0], device=token_indices.device)
        
        # Concatenate and sort for predictable order
        final_indices = torch.cat([cls_token_index, token_indices]).sort().values
        
        return final_indices

    def forward_pruned(self, x: torch.Tensor, top_k_indices: torch.Tensor):
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        print(f"PRUNING: Tensor shape before: {x.shape[1]} tokens")
        x = torch.index_select(x, dim=1, index=top_k_indices.to(x.device))
        print(f"PRUNING: Tensor shape after: {x.shape[1]} tokens")

        # Process the reduced set of tokens through the transformer blocks
        # We disable UCB logic since pruning is already done
        for i, block in enumerate(self.blocks):
            x, _ = block(
                x, 
                counter=99999,
                ucb_enabled=False,  # Disabilita UCB durante inferenza pruned
                ucb_count_score=None
            )
            
        # Final norm and classification (CLS token is still at index 0)
        x = self.norm(x)
        logits = self.head(x[:, 0])
        
        return logits

    def get_pruning_stats(self):
        """Returns statistics about current pruning configuration and state."""
        stats = {
            'num_layers': len(self.blocks),
            'num_heads': self.blocks[0].attn.num_heads,
            'total_patches': self.pos_embed.shape[1],
        }
        
        # Add per-layer pruning configuration
        for i, block in enumerate(self.blocks):
            if hasattr(block.attn, 'keep_ratio') and block.attn.keep_ratio is not None:
                stats[f'layer_{i}_keep_ratio'] = block.attn.keep_ratio
                stats[f'layer_{i}_kept_patches'] = int(
                    (self.pos_embed.shape[1] - (1 if block.attn.exclude_cls else 0)) * block.attn.keep_ratio
                )
            elif hasattr(block.attn, 'k') and block.attn.k is not None:
                total_patches = self.pos_embed.shape[1] - (1 if block.attn.exclude_cls else 0)
                stats[f'layer_{i}_keep_ratio'] = block.attn.k / total_patches
                stats[f'layer_{i}_kept_patches'] = block.attn.k
                
        # Add UCB statistics
        stats['avg_patch_counts'] = self.ucb_count_scores.mean(dim=(0, 1)).cpu().numpy()
        stats['min_patch_count'] = self.ucb_count_scores.min().item()
        stats['max_patch_count'] = self.ucb_count_scores.max().item()
        
        return stats