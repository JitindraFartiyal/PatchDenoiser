import time
import torch
from torch import nn
from PatchUp.patch_encoder import L32PatchEncoder, L128PatchEncoder, L512PatchEncoder
from decoder import Decoder
from utilities import get_model_stats

def reconstruct_feature_map(patch_latents, B, n_h, n_w):
    # patch_latents: [B*n_patches, C, h, w]
    # n_h, n_w = number of patches along H and W
    C, h, w = patch_latents.shape[1:]
    patch_latents = patch_latents.view(B, n_h, n_w, C, h, w)
    # permute to [B, C, n_h*h, n_w*w]
    patch_latents = patch_latents.permute(0, 3, 1, 4, 2, 5).contiguous()
    feature_map = patch_latents.view(B, C, n_h*h, n_w*w)
    return feature_map

def get_patches(x, patch_size):

    b, c, h, w = x.size()
    # Unfold height and width
    patches = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    # patches.shape -> [b, c, n_h, n_w, patch_size, patch_size]
    n_h, n_w = patches.size(2), patches.size(3)

    # Move c to after patches and combine n_h*n_w into num_patches
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    # patches.shape -> [b, n_h, n_w, c, patch_size, patch_size]

    patches = patches.view(b, n_h * n_w, c, patch_size, patch_size)  # [b, num_patches, c, patch_size, patch_size]

    return patches.squeeze(2)

class SGatedFusion(nn.Module):
    def __init__(self, in_channels_low, in_channels_high, out_channels):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels_low, out_channels, kernel_size=1, bias=False),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(out_channels + in_channels_high, out_channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, low_feat, high_feat):
        low_proj = self.proj(low_feat)

        concat = torch.cat([low_proj, high_feat], dim=1)
        gate_concat = self.gate(concat)

        g = self.sigmoid(gate_concat)

        fused = g * low_proj + (1 - g) * high_feat

        # return fused
        return gate_concat

class GatedFusion(nn.Module):
    def __init__(self, in_channels_low, in_channels_high, out_channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(out_channels + in_channels_high, out_channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, low_feat, high_feat):

        concat = torch.cat([low_feat, high_feat], dim=1)
        gate_concat = self.gate(concat)
        g = self.sigmoid(gate_concat)

        fused = g * low_feat + (1 - g) * high_feat

        # return fused
        return gate_concat


class PatchUp(nn.Module):
    def __init__(self, img_size=512, patch_sizes=None):
        super().__init__()
        if patch_sizes is None:
            patch_sizes = [32, 128, 512]
        self.patch_sizes = patch_sizes
        self.img_size = img_size

        le_32_oc, le_128_oc, le_512_oc = 8, 16, 24
        le_32_ld, le_128_ld, le_512_ld = 8, 16, 24

        self.patch_encoders = nn.ModuleList([
            L32PatchEncoder(out_channels=le_32_oc, latent_dim=le_32_ld),
            L128PatchEncoder(out_channels=le_128_oc, latent_dim=le_128_ld),
            L512PatchEncoder(out_channels=le_512_oc, latent_dim=le_512_ld),
        ])

        self.decoder = Decoder(in_channels=le_512_ld)

        self.l32_128_fusion = SGatedFusion(le_32_ld, le_128_ld, le_128_ld)
        self.l32_512_fusion = SGatedFusion(le_32_ld, le_512_ld, le_512_ld)
        self.l128_512_fusion = SGatedFusion(le_128_ld, le_512_ld, le_512_ld)
        self.l128_512_joint_fusion = SGatedFusion(le_128_ld, le_512_ld, le_512_ld)

        self.f1 = GatedFusion(le_512_ld, le_512_ld, le_512_ld)
        self.f2 = GatedFusion(le_512_ld, le_512_ld, le_512_ld)

    def forward(self, x):
        b, c, h, w = x.size()

        patches_embedding = []
        for patch_size, encoder in zip(self.patch_sizes, self.patch_encoders):
            curr_patch = get_patches(x, patch_size=patch_size)  # [b, num_patches, c, patch_size, patch_size]
            curr_patch = curr_patch.view(-1, c, patch_size, patch_size)
            curr_patch_embedding = encoder(curr_patch)
            # print(curr_patch.shape, curr_patch_embedding.shape)
            curr_patch_embedding = reconstruct_feature_map(curr_patch_embedding, b, self.img_size // patch_size, self.img_size // patch_size)
            patches_embedding.append(curr_patch_embedding)

        # print(patches_embedding[0].size(), patches_embedding[1].size(), patches_embedding[2].size())
        l32_128_fused = self.l32_128_fusion(patches_embedding[0], patches_embedding[1])
        l32_512_fused = self.l32_512_fusion(patches_embedding[0], patches_embedding[2])
        l128_512_fused = self.l128_512_fusion(patches_embedding[1], patches_embedding[2])

        l128_512_joint_fused = self.l128_512_joint_fusion(l32_128_fused, patches_embedding[2])

        fused1 = self.f1(l32_512_fused, l128_512_fused)
        fused2 = self.f2(fused1, l128_512_joint_fused)
        out = self.decoder(fused2)

        return out


if __name__ == '__main__':
    device = torch.device('cuda')
    input = torch.randn(1, 1, 512, 512).to(device)
    start_time = time.time()
    model = PatchUp().to(device)
    output = model(input)
    print(output.size(), time.time() - start_time)


    # get_model_stats(model, dummy_input=torch.randn(1, 1, 512, 512).to(device))