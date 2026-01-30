"""
Training & Inference Engine
---------------------------
This module provides the core logic for training, evaluating, and generating text.

Key Features:
- Autoregressive Generation: Implements temperature scaling and Top-K/Top-P (Nucleus) sampling for creative yet coherent text.
- KV-Cache Optimized Inference: Uses a stateful generation loop that caches Key and Value tensors to achieve O(1) incremental decoding speed.
- High-Performance Training: Optimizes Sparse MoE and Dual-Stream architectures using Automatic Mixed Precision (AMP) and Gradient Clipping for stability.
- Comprehensive Validation: Tracks both Cross-Entropy loss and Perplexity across training epochs to monitor linguistic convergence.
"""

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from functools import partial
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


"""
Text Generation Utilities
-------------------------
These functions manage the 'Autoregressive' loop that allows the model to speak.
It handles the conversion between text and tensors and implements sampling
strategies like Temperature and Top-K to control the model's creativity.
"""
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())


def top_k_top_p_filtering(logits, top_k=None, top_p=0.9):
    """
    Apply top-k and/or top-p (nucleus) filtering to logits.

    Args:
        logits: (B, V) tensor of unnormalized log-probs
        top_k:  int or None
        top_p:  float in (0,1) or None

    Returns:
        filtered_logits: (B, V) with some positions set to -inf
    """
    B, V = logits.shape

    # ----- Top-k -----
    if top_k is not None and top_k > 0 and top_k < V:
        top_k = int(top_k)
        top_logits, top_idx = torch.topk(logits, top_k, dim=-1)   # (B, top_k)
        filtered = torch.full_like(logits, float('-inf'))         # (B, V)
        filtered.scatter_(dim=-1, index=top_idx, src=top_logits)  # keep only top-k
        logits = filtered

    # ----- Top-p (nucleus) -----
    if top_p is not None and 0.0 < top_p < 1.0:
        # Sort by logit descending
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)  # (B, V)
        sorted_probs = F.softmax(sorted_logits, dim=-1)                              # (B, V)
        cumulative_probs = sorted_probs.cumsum(dim=-1)                               # (B, V)

        # Mask tokens where cumulative prob > top_p
        sorted_mask = cumulative_probs > top_p                                       # (B, V)
        # Always keep at least one token
        sorted_mask[..., 0] = False

        # Set masked logits to -inf
        sorted_logits = sorted_logits.masked_fill(sorted_mask, float('-inf'))

        # Unsort back to original order
        logits = torch.full_like(logits, float('-inf'))
        logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)

    return logits


def generate(
    model,
    idx,
    max_new_tokens,
    context_size,
    use_cache: bool = True,
    top_k: int | None = None,
    top_p: float | None = None,
    temperature: float = 0.0,
    eos_id: int | None = None
):
    """
    Inference-time generation with:
      - optional KV cache
      - top-k + top-p + temperature sampling

    idx: (B, T_start)
    """
    if use_cache:
        model.reset_cache()  # clear KV cache for a fresh sequence

    B = idx.size(0)

    for step in range(max_new_tokens):
        # Decide what to feed the model
        if use_cache:
            if step == 0:
                # Prefill: full prompt (cropped to context_size)
                idx_cond = idx[:, -context_size:]      # (B, T_prompt)
            else:
                # Incremental: only last generated token
                idx_cond = idx[:, -1:]                 # (B, 1)
        else:
            # No cache: always feed last context window
            idx_cond = idx[:, -context_size:]          # (B, T_ctx)

        with torch.no_grad():
            logits = model(idx_cond, use_cache=use_cache)[:, -1, :]  # (B, V)

        # Deterministic (greedy) path
        if temperature <= 0.0:
            next_token_id = torch.argmax(logits, dim=-1, keepdim=True)  # (B, 1)
        else:
            # Scale by temperature
            logits = logits / temperature

            # Numerical stability trick
            logits = logits - logits.max(dim=-1, keepdim=True).values

            # Apply top-k and/or top-p
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

            # Probabilities
            probs = torch.softmax(logits, dim=-1)                      # (B, V)

            # Sample from distribution
            next_token_id = torch.multinomial(probs, num_samples=1)    # (B, 1)

        if next_token_id == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break
        # Append sampled token to full sequence
        idx = torch.cat((idx, next_token_id), dim=-1)                  # (B, T+1)

    return idx


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate(
            model=model, idx=encoded,
            max_new_tokens=25, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()
    
    

"""
Training Utilities & Optimization Loop
--------------------------------------
This module contains the core training engine. It handles batch-wise loss 
calculations, model evaluation, and the main training loop (train_model_simple).    
# -------------------------------------------------------------------
"""
# -------------------------------------------------------------------
# Calc loss helpers
# -------------------------------------------------------------------
def calc_loss_batch(input_batch, target_batch, model, loss_func, device):
    """
    Compute loss for a single batch.

    Expects:
      - input_batch: (B, T) token IDs
      - target_batch: (B, T) token IDs (next-token targets)
      - model(input_batch): (B, T, V) logits

    Flattens to (B*T, V) vs (B*T,) for cross-entropy.
    """
    # Move to device (non_blocking requires DataLoader pin_memory=True)
    input_batch = input_batch.to(device, non_blocking=True)
    target_batch = target_batch.to(device, non_blocking=True)

    logits = model(input_batch)  # (B, T, V)
    loss = loss_func(
        logits.flatten(0, 1),    # (B*T, V)
        target_batch.flatten()   # (B*T,)
    )
    return loss


def calc_loss_loader(data_loader, model, device, loss_func, num_batches=None):
    """
    Average loss over 'num_batches' from a data loader.
    Does NOT track gradients (should be called under torch.no_grad()).
    """
    if len(data_loader) == 0:
        return float("nan")

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    total_loss = 0.0
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        batch_loss = calc_loss_batch(input_batch, target_batch, model, loss_func, device)
        total_loss += batch_loss.item()
    return total_loss / num_batches


# -------------------------------------------------------------------
# Evaluation + sample generation
# -------------------------------------------------------------------
def evaluate_model(
    model,
    train_loader,
    val_loader,
    device,
    eval_iter,
    loss_func=F.cross_entropy
):
    """
    Evaluate model on:
      - first 'eval_iter' batches of train_loader
      - first 'eval_iter' batches of val_loader

    Returns scalar floats: (train_loss, val_loss).
    """
    was_training = model.training
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader,
            model,
            device=device,
            loss_func=loss_func,
            num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader,
            model,
            device=device,
            loss_func=loss_func,
            num_batches=eval_iter
        )
    if was_training:
        model.train()
    return train_loss, val_loss

# -------------------------------------------------------------------
# MAIN TRAINING LOOP CODE
# -------------------------------------------------------------------
def train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs, eval_freq, eval_iter, start_context, tokenizer,
    cfg=None,                       # Dictionary containing the model's 'DNA' (architecture settings)
    checkpoint_path="best_model.pth", # Path to save the 'Val Loss Champion' (lowest error model)
    final_save_path="final_model.pth",# Path to save the 'Safety' version when training is 100% done
    loss_func=partial(F.cross_entropy, label_smoothing=0.0), # The math formula for 'error'
    use_amp=True,                   # Toggle: True uses 'Mixed Precision' to speed up training by 2-3x
    temperature=0.7,                # Creativity: Higher (1.0+) is wilder; Lower (0.1) is more robotic
    top_p=0.9,                      # Coherence: Only picks from words making up top 90% of probability
    start_epoch=0,                  # The epoch number to start at (useful for resuming training)
    global_step=0,                  # Counter for total batches seen across the model's lifetime
    tokens_seen=0,                  # Counter for the total volume of words processed
    best_val=float("inf"),          # The 'High Score' (lowest loss) to beat for checkpointing
    do_live_sample=True,            # Toggle: Generate text snippets during training to see progress
    do_save_best=True               # Toggle: Automatically save files when a better model is found
):
    """
    Standardized training loop optimized for Sparse MoE and Dual-Stream architectures.
    Features: Mixed Precision (AMP), Gradient Clipping, Cosine Scheduling, and Live Sampling.
    """
    train_losses, val_losses, track_tokens_seen = [], [], []

    # 1. LEARNING RATE SCHEDULER: Implements Cosine Decay to ensure smooth convergence
    total_steps = num_epochs * len(train_loader)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    # 2. GRADIENT SCALER: Prevents numerical underflow when using float16/bfloat16
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp and device.type == "cuda")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for input_batch, target_batch in pbar:
            # 3. STABLE DATA TRANSFER: Synchronous moves to prevent CPU-GPU deadlocks
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            # Optimized gradient clearing (faster than zero_grad())
            optimizer.zero_grad(set_to_none=True)

            # 4. AUTO-MIXED PRECISION (AMP): Dynamically selects the best dtype for the GPU
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            with torch.amp.autocast('cuda', enabled=use_amp, dtype=dtype):
                logits = model(input_batch)
                # Flattening for CrossEntropy: (Batch * Seq, Vocab)
                loss = loss_func(logits.flatten(0, 1), target_batch.flatten())

            # 5. BACKWARD PASS & STABILITY: Essential for Mixture of Experts (MoE)
            if use_amp and device.type == "cuda":
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                # GRADIENT CLIPPING: Prevents 'Exploding Gradients' from unstable MoE routing
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
            else:
                # Fallback for CPU or non-AMP training
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # 6. STEP UPDATES: Update LR and global progress tracking
            scheduler.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            pbar.set_postfix(loss=f"{loss.item():.4f}")

            # 7. PERIODIC EVALUATION: Snapshot performance on unseen data
            if global_step % eval_freq == 0:

                # Validation pass (no_grad)
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter, loss_func)

                # Checkpointing: Save the weights that achieve the lowest validation error
                if do_save_best and val_loss < best_val:
                    best_val = val_loss
                    
                    # UPDATED: Create a Comprehensive Checkpoint (Weights + Blueprint)
                    checkpoint = {
                        "model_state_dict": model.state_dict(),
                        "config": cfg,              # The architecture blueprint
                        "val_loss": val_loss,
                        "global_step": global_step,
                        "tokens_seen": tokens_seen
                    }
                    torch.save(checkpoint, checkpoint_path)
                    print(f"💾 Comprehensive best model saved: {checkpoint_path}")

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

                print(f"\n✨ Step {global_step}: Train {train_loss:.3f}, Val {val_loss:.3f} | Best: {best_val:.3f}")

                # 8. QUALITATIVE LIVE SAMPLING: Monitor linguistic recovery during training
                if do_live_sample:
                    print(f"📝 Live Sample (Step {global_step}):")
                    model.eval()
                    with torch.no_grad():
                        generate_and_print_sample(model, tokenizer, device, start_context)
                    model.train() # Resume training mode
                    print("-" * 30)

        # END OF EPOCH SAMPLING: Always run a sample at the end of an epoch regardless of flags
        print(f"🏁 End of Epoch {epoch+1} qualitative check:")
        model.eval()
        with torch.no_grad():
            generate_and_print_sample(model, tokenizer, device, start_context)
        model.train()
        print("-" * 30)

    # 9. FINAL COMPREHENSIVE SAVE: One additional save before the function ends
    print(f"🚀 Training complete. Saving final state to {final_save_path}...")
    final_checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "global_step": global_step,
        "tokens_seen": tokens_seen,
        "final_val_loss": val_losses[-1] if val_losses else None
    }
    torch.save(final_checkpoint, final_save_path)

    return train_losses, val_losses, track_tokens_seen


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    """
    Visualizes the model's learning progress by plotting training and validation loss.
    
    This function generates a dual-axis plot comparing loss against both the 
    number of training epochs (passes through the data) and the total volume 
    of tokens processed by the model.
    """
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs (primary x-axis)
    # The solid line shows how well the model is learning the training data.
    # The dash-dot (-.) line helps us spot overfitting on the validation set.
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    
    # Ensure the bottom axis only shows whole numbers for epochs
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Create a second x-axis for tokens seen (top x-axis)
    # This shares the same y-axis but allows us to see performance 
    # relative to the raw amount of data processed.
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot to align ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Prevents labels from being cut off
    
    # Save as a PDF for a high-quality, vector-based visual for the final report
    plt.savefig("loss-plot.pdf")
    plt.show()