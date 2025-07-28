import torch
import torch.nn as nn
import torch.nn.functional as F

# Learnable Daubechies Wavelet Transform (db2-style)
class LearnableDaubechiesTransform(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Base filters for Daubechies 4 (db2)
        # h0 (low-pass): smoothing filter
        base_h0 = torch.tensor([0.48296, 0.83652, 0.22414, -0.12941])
        # h1 (high-pass): detail filter (quadrature mirror filter relationship)
        base_h1 = torch.tensor([-0.12941, -0.22414, 0.83652, -0.48296])

        # Make filters learnable parameters, expanding to 'dim' features
        # [1, kernel_size, 1] * [1, 1, dim] -> [1, kernel_size, dim]
        self.h0 = nn.Parameter(base_h0.view(1, 4, 1) * torch.ones(1, 4, dim))
        self.h1 = nn.Parameter(base_h1.view(1, 4, 1) * torch.ones(1, 4, dim))

    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.shape

        # Padding for valid convolution/unfolding with kernel_size=4, stride=2
        # We need (T_padded - 4) to be non-negative and divisible by 2.
        # Minimum T for output length 1 is 4. T_padded must be at least 4.
        # output_length = floor((T_padded - kernel_size) / stride) + 1
        # output_length = floor((T_padded - 4) / 2) + 1
        # The current padding `(2 - (T - 4) % 2) % 2` ensures `T_padded - 4` is even if T >= 4.
        # If T < 4, this padding might not make T_padded >= 4, resulting in 0 output length.
        # Let's use F.pad with 'reflect' or 'replicate' for boundary handling,
        # and pad enough so T_padded >= 4 and (T_padded - 4) % 2 == 0.
        # A simpler pad that works for kernel 4, stride 2 is padding by 3 on the left and adjust right.
        # Let's stick to the unfold approach but ensure sufficient padding.
        # The required padding for output length L = ceil(T/2) roughly is kernel_size - stride = 4 - 2 = 2
        # Pad on the right to handle sequence end.
        # A common strategy for wavelets is periodic or reflection padding.
        # For simplicity with unfold, let's pad the end to make length suitable.
        # Target length T_padded such that T_padded >= T and (T_padded - 4) is even and non-negative.
        min_T_padded = max(T, 4) # Ensure length is at least 4
        if (min_T_padded - 4) % 2 != 0:
             min_T_padded += 1 # Ensure (T_padded - 4) is even

        pad_len = min_T_padded - T
        if pad_len > 0:
             # Pad on the right (end of sequence)
             pad = torch.zeros(B, pad_len, D, device=x.device, dtype=x.dtype)
             x = torch.cat([x, pad], dim=1)
             T_padded = x.shape[1]
        else:
            T_padded = T # No padding needed if T is already suitable

        # Perform unfolding for efficient convolution
        # Need shape [B, D, T_padded, 1] for F.unfold with kernel (4, 1)
        x_unfold = F.unfold(x.transpose(1, 2).unsqueeze(-1), kernel_size=(4, 1), stride=2)
        # x_unfold shape: [B, D * kernel_size, num_windows] -> [B, D * 4, (T_padded - 4) // 2 + 1]

        # Reshape and permute to apply filters element-wise across D
        # [B, D*4, L] -> [B, D, 4, L] -> [B, L, 4, D] where L is num_windows
        num_windows = x_unfold.shape[-1]
        x_unfold = x_unfold.view(B, D, 4, num_windows).permute(0, 3, 2, 1) # [B, num_windows, 4, D]

        # Apply learnable filters: sum( [B, L, 4, D] * [1, 4, D], dim=2 ) -> [B, L, D]
        low = torch.sum(x_unfold * self.h0, dim=2) # Approximation coefficients
        high = torch.sum(x_unfold * self.h1, dim=2) # Detail coefficients

        # low, high shape: [B, num_windows, D] where num_windows = (T_padded - 4) // 2 + 1
        return low, high

# Multi-Scale Wavelet
class MultiScaleDaubechies(nn.Module):
    def __init__(self, dim, levels=3):
        super().__init__()
        self.levels = levels # Store levels
        # Use LearnableDaubechiesTransform at each level
        self.transforms = nn.ModuleList([LearnableDaubechiesTransform(dim) for _ in range(levels)])

    def forward(self, x):
        # x: [B, T, D]
        # Store original sequence length before any transformations
        original_T = x.shape[1]
        details = []
        approx = x # Initialize approx as the input

        # Apply transform level by level
        for l in range(self.levels):
            # The LearnableDaubechiesTransform handles padding internally for the current level input
            approx, d = self.transforms[l](approx)
            # Store detail coefficients
            details.append(d)

        # Return final approximation and all detail coefficients
        # approx: [B, T_approx, dim]
        # details: list of [B, T_d_l, dim] for l=1...levels
        # Note: T_approx and T_d_l are reduced lengths due to downsampling
        return original_T, approx, details # --- Modification: Return original_T ---

# FeedForward Network (No change needed)
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )
    def forward(self, x):
        return self.net(x)

# Multi-Head Wavelet Encoder Block
class MultiHeadDaubechiesBlock(nn.Module):
    def __init__(self, dim, ffn_dim, levels=3, heads=4):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.levels = levels # Store levels

        # Use MultiScaleDaubechies for each head
        self.transforms = nn.ModuleList([MultiScaleDaubechies(self.head_dim, levels) for _ in range(heads)])

        # --- Modification: Keep norms and proj/ffn as they will operate on the combined output ---
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)
        self.ffn = FeedForward(dim, ffn_dim)

        # --- Modification: No extra parameters needed if summing upsampled components ---
        # The combination happens by upsampling and summing, preserving the head_dim for proj input.

    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.shape
        original_T = T # Store original sequence length

        # Apply LayerNorm before the mixer
        x_norm = self.norm1(x)

        # Reshape for multi-head processing
        # [B, T, D] -> [B, T, heads, head_dim] -> [B, heads, T, head_dim]
        x_heads = x_norm.view(B, T, self.heads, self.head_dim).transpose(1, 2)

        outputs = [] # To collect combined results from each head
        for h in range(self.heads):
            xi = x_heads[:, h] # Input for head h: [B, T, head_dim]

            # --- Modification: Get original_T, approximation, and details ---
            # Apply multi-level Daubechies transform
            head_original_T, approx, details = self.transforms[h](xi)

            # --- Modification: Combine approximation and detail coefficients by upsampling and summing ---
            # Collect all components: the final approximation and details from all levels
            # approx: [B, T_approx, head_dim]
            # details: list of [B, T_d_l, head_dim]
            all_components = [approx] + details

            # Upsample each component back to the original sequence length T and sum them
            # Initialize tensor to accumulate upsampled components
            combined_output = torch.zeros(B, original_T, self.head_dim, device=x.device, dtype=x.dtype)

            for component in all_components:
                 # F.interpolate expects input shape [B, C, spatial_dim]
                 # Our component shape is [B, Time, Features], so transpose to [B, Features, Time]
                 # Interpolate along the time dimension to size original_T
                 # Then transpose back to [B, Time, Features]
                 upsampled_component = F.interpolate(
                     component.transpose(1, 2),
                     size=original_T, # Upsample to the original input sequence length
                     mode='linear', # Use linear interpolation for 1D sequences
                     align_corners=False # Recommended for non-image data
                 ).transpose(1, 2)

                 # Sum the upsampled components
                 combined_output += upsampled_component

            outputs.append(combined_output) # combined_output shape: [B, original_T, head_dim]

        # Concatenate the combined outputs from all heads along the feature dimension
        x_mixed = torch.cat(outputs, dim=2)  # [B, original_T, D]

        # Apply the linear projection after combining heads
        x_mixed = self.proj(x_mixed) # [B, original_T, D]

        # --- Modification: Residual connection using the *original* input x ---
        # Add the mixed result to the original input (before normalization)
        x = x + x_mixed # [B, T, D] + [B, T, D] -> [B, T, D]

        # Apply LayerNorm before FFN
        x_norm_ffn = self.norm2(x)

        # Apply FFN
        x_ffn = self.ffn(x_norm_ffn) # [B, T, D]

        # --- Modification: Residual connection after FFN ---
        # Add the FFN output to the result after the mixing step
        x = x + x_ffn # [B, T, D] + [B, T, D] -> [B, T, D]

        # The output shape is [B, T, D], matching the input shape
        return x

# Patch Embedding (No change needed - image specific, but kept for context)
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=192):
        super().__init__()
        # Ensure img_size is divisible by patch_size for integer grid_size
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2 # This becomes the sequence length T for images
        # Convolution to get patches and project to embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x) # [B, embed_dim, H/patch_size, W/patch_size]
        # Flatten spatial dimensions and transpose to get [B, num_patches, embed_dim] (which is [B, T, D])
        x = x.flatten(2).transpose(1, 2)
        return x

# Final ViT Model (Minimal changes needed to accommodate the block's consistent I/O shape)
class ViT_Daubechies(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 num_classes=10, embed_dim=192, depth=6, ffn_dim=768,
                 levels=2, heads=4):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches # Number of patches is the sequence length

        # Add CLS token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Positional embedding needs to cover patch tokens + CLS token
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, embed_dim))
        self.dropout = nn.Dropout(0.1)

        # Stack the modified MultiHeadDaubechiesBlock blocks
        # These blocks now preserve the sequence length
        self.blocks = nn.ModuleList([
            MultiHeadDaubechiesBlock(embed_dim, ffn_dim, levels, heads) for _ in range(depth)
        ])

        # Final normalization and classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights (optional, but good practice)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights) # Initialize other layers

    def _init_weights(self, m):
        # Standard initialization for Linear and Conv2d layers
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        # LayerNorm and Parameter initializations are handled separately or use defaults

    def forward(self, x):
        # x: [B, C, H, W] (image input)
        B = x.size(0)

        # Patch embedding: [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)

        # Add CLS token: [B, num_patches, embed_dim] -> [B, 1 + num_patches, embed_dim]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional embedding
        x = x + self.pos_embed[:, :x.size(1)]

        # Apply dropout
        x = self.dropout(x)

        # Pass through stacked Daubechies Mixer Blocks
        # Each block takes [B, T, D] and outputs [B, T, D]
        for blk in self.blocks:
            x = blk(x)

        # Final normalization
        x = self.norm(x)

        # Take the CLS token representation for classification
        cls_token_output = x[:, 0] # Shape [B, embed_dim]

        # Classification head
        out = self.head(cls_token_output) # Shape [B, num_classes]

        return out

# === Test ===
if __name__ == "__main__":
    print("Testing MultiHeadDaubechiesBlock with time series input...")

    # Example usage for time series (stacking blocks)
    # Input shape [B, T, D]
    batch_size = 16
    sequence_length = 12 # T
    feature_dimension = 96 # D

    # Define block parameters
    block_dim = feature_dimension
    block_ffn_dim = block_dim * 4 # Typical expansion ratio
    block_num_heads = 4
    block_levels = 2 # As in your example

    # Instantiate a single MultiHeadDaubechiesBlock
    daubechies_block = MultiHeadDaubechiesBlock(
        dim=block_dim,
        ffn_dim=block_ffn_dim,
        levels=block_levels,
        heads=block_num_heads
    )

    # Create a dummy time series input
    dummy_ts_input = torch.randn(batch_size, sequence_length, feature_dimension)
    print(f"Input shape: {dummy_ts_input.shape}")

    # Pass through the block
    output = daubechies_block(dummy_ts_input)
    print(f"Output shape: {output.shape}")

    # Verify the output shape matches the input shape for stacking
    assert output.shape == dummy_ts_input.shape, "Output shape does not match input shape!"
    print("Shape matches! Block is stackable.")

    print("\nTesting ViT_Daubechies (Image example - uses the blocks internally)...")
    # Test the full ViT_Daubechies class (image input)
    # Using parameters from your test case
    vit_model = ViT_Daubechies(img_size=224, patch_size=16, in_chans=3,
                               num_classes=10, embed_dim=192, depth=6,
                               ffn_dim=768, levels=2, heads=4)
    dummy_img_input = torch.randn(4, 3, 224, 224)
    print(f"ViT Input shape: {dummy_img_input.shape}")

    vit_output = vit_model(dummy_img_input)
    print(f"ViT Output shape: {vit_output.shape}")  # Expected: [4, 10]
    assert vit_output.shape == torch.Size([4, 10])

    # Check parameter count
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nParameter count of MultiHeadDaubechiesBlock: {count_parameters(daubechies_block)}")
    print(f"Parameter count of ViT_Daubechies: {count_parameters(vit_model)}")