import os, torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import random
from skimage.metrics import structural_similarity as ssim
import cv2

# ===== 路径配置  =====
ckpt_path = "./model/v1-5-pruned.ckpt"
clip_path = "./model/clip-vit-large-patch14"
# 训练好的VAE路径
trained_vae_path = "./sd15_ir_vae_lpips/vae_best_lpips"
# 微调好的LoRA权重路径
lora_weights_path = "./sd15_ir_lora_with_trained_vae/lora_unet_best.pt"
# 测试数据集路径
test_data_dir = "./data/raw_5k" 
output_dir = "./infrared_reconstruction_ssim_test"
os.makedirs(output_dir, exist_ok=True)

# 推理参数 - 适配512x512
image_size = 512  # 修改为512x512
num_inference_steps = 30
guidance_scale = 7.5
seed = 42
device = "cuda"

# 设置种子以确保可重复性
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

print("=== 512x512红外特征重建测试与SSIM评估 ===")
print(f"VAE路径: {trained_vae_path}")
print(f"LoRA路径: {lora_weights_path}")
print(f"测试数据路径: {test_data_dir}")
print(f"图像尺寸: {image_size}")

# ===== 1. 加载基础模型 =====
print("\n正在加载基础模型...")

try:
    # 手动加载checkpoint
    print("正在加载checkpoint文件...")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # 构建UNet
    print("正在构建UNet...")
    unet = UNet2DConditionModel(
        sample_size=image_size // 8,  
        in_channels=4,
        out_channels=4,
        layers_per_block=2,
        block_out_channels=[320, 640, 1280, 1280],
        down_block_types=[
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D", 
            "CrossAttnDownBlock2D",
            "DownBlock2D"
        ],
        up_block_types=[
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D"
        ],
        cross_attention_dim=768,
        attention_head_dim=8,
    )
    
    # 加载UNet权重
    unet_dict = unet.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() if k in unet_dict}
    unet_dict.update(pretrained_dict)
    unet.load_state_dict(unet_dict)
    print("UNet加载完成")
    
    # 构建VAE
    print("正在构建VAE...")
    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=["DownEncoderBlock2D"] * 4,
        up_block_types=["UpDecoderBlock2D"] * 4,
        block_out_channels=(128, 256, 512, 512),
        latent_channels=4,
        sample_size=image_size,  # 设置为512
    )
    
    # 加载VAE权重
    vae_dict = vae.state_dict()
    vae_pretrained_dict = {k: v for k, v in state_dict.items() if k in vae_dict}
    vae_dict.update(vae_pretrained_dict)
    vae.load_state_dict(vae_dict)
    print("VAE加载完成")
    
    # 加载CLIP组件
    print("正在加载CLIP组件...")
    tokenizer = CLIPTokenizer.from_pretrained(clip_path, local_files_only=True)
    text_encoder = CLIPTextModel.from_pretrained(clip_path, local_files_only=True)
    print("CLIP组件加载完成")
    
    # 创建调度器
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
    )
    
    pipe = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
    )
    
    print("基础管道构建完成")
    
except Exception as e:
    print(f"基础模型加载失败: {e}")
    exit(1)

# ===== 2. 加载训练好的VAE =====
print(f"\n正在加载训练好的VAE: {trained_vae_path}...")
try:
    trained_vae = AutoencoderKL.from_pretrained(trained_vae_path, torch_dtype=torch.float16).to(device)
    pipe.vae = trained_vae
    print(f"训练VAE配置 - latent_channels: {trained_vae.config.latent_channels}, sample_size: {trained_vae.config.sample_size}")
    print("✅ 训练好的VAE加载成功")
except Exception as e:
    print(f"❌ 加载训练VAE失败: {e}")
    print("使用原始VAE进行测试...")
    trained_vae = pipe.vae

# ===== 3. 加载LoRA权重 =====
print(f"\n正在加载LoRA权重: {lora_weights_path}...")

# 检查LoRA权重结构
lora_state_dict = torch.load(lora_weights_path, map_location="cpu")
print(f"LoRA权重键数量: {len(lora_state_dict.keys())}")

# 检查是否包含LoRA特有的键名
lora_specific_keys = [k for k in lora_state_dict.keys() if any(x in k for x in ['down', 'up'])]
print(f"LoRA特有键数量: {len(lora_specific_keys)}")
if lora_specific_keys:
    print(f"前5个LoRA键: {lora_specific_keys[:5]}")

# 检查权重是否真的包含了LoRA参数
has_lora_params = any('lora' in k.lower() or ('down' in k and 'weight' in k) or ('up' in k and 'weight' in k) for k in lora_state_dict.keys())
print(f"是否包含LoRA参数: {has_lora_params}")

# 尝试加载权重
try:
    # 检查键名匹配情况
    unet_keys = set(pipe.unet.state_dict().keys())
    lora_keys = set(lora_state_dict.keys())
    
    matching_keys = unet_keys.intersection(lora_keys)
    print(f"匹配的键数量: {len(matching_keys)} (总共有{len(unet_keys)}个UNet键)")
    
    # 尝试加载
    pipe.unet.load_state_dict(lora_state_dict, strict=False)
    print("LoRA权重加载完成")
    
except Exception as e:
    print(f"LoRA权重加载失败: {e}")

# 移动到GPU并设置为eval模式
pipe = pipe.to(device, dtype=torch.float16)
pipe.unet.eval()
pipe.vae.eval()
pipe.text_encoder.eval()

print(f"\n模型配置:")
print(f"- VAE sample_size: {pipe.vae.config.sample_size}")
print(f"- VAE latent_channels: {pipe.vae.config.latent_channels}")
print(f"- UNet sample_size: {pipe.unet.config.sample_size}")

# ===== 4. SSIM计算函数 =====
def calculate_ssim(img1, img2):
    """计算两张图像的SSIM值"""
    # 转换为numpy数组
    if isinstance(img1, Image.Image):
        img1 = np.array(img1.convert('L'))  # 转为灰度
    if isinstance(img2, Image.Image):
        img2 = np.array(img2.convert('L'))  # 转为灰度
    
    # 确保图像维度一致
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # 计算SSIM
    ssim_value = ssim(img1, img2, data_range=255)
    return ssim_value

def calculate_mse_psnr(img1, img2):
    """计算MSE和PSNR"""
    if isinstance(img1, Image.Image):
        img1 = np.array(img1.convert('L')).astype(np.float64)
    if isinstance(img2, Image.Image):
        img2 = np.array(img2.convert('L')).astype(np.float64)
    
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        psnr = float('inf')
    else:
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    return mse, psnr

# ===== 5. 测试数据集 =====
class TestDataset(Dataset):
    def __init__(self, data_dir, transform):
        self.image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        self.transform = transform
        print(f"找到 {len(self.image_paths)} 张测试图像")
        
    def __len__(self): return min(10, len(self.image_paths))  # 只测试前10张
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        return {"pixel_values": self.transform(img), "path": img_path}

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),  # 512x512
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

test_dataset = TestDataset(test_data_dir, transform)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# ===== 6. 生成测试 =====
print(f"\n=== 开始生成测试 ===")

# 设置调度器
pipe.scheduler.set_timesteps(num_inference_steps)

# 测试不同类型的提示
test_prompts = [
    "",  # 空提示 - 应该生成红外特征
    "infrared thermal image",  # 红外相关提示
    "thermal imaging",  # 热成像提示
    "heat signature",  # 热信号提示
]

results = []

for i, prompt in enumerate(test_prompts):
    print(f"\n生成测试 {i+1}: '{prompt or '无提示'}'")
    
    generator = torch.Generator(device=device).manual_seed(seed + i)
    
    with torch.no_grad():
        generated_img = pipe(
            prompt=prompt if prompt else " ",
            height=image_size,
            width=image_size,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="pil"
        ).images[0]
    
    # 保存生成的图像
    gen_filename = f"generated_{i+1}_{prompt.replace(' ', '_')[:20] if prompt else 'empty'}_512.png"
    gen_filepath = os.path.join(output_dir, gen_filename)
    generated_img.save(gen_filepath)
    print(f"生成图像已保存: {gen_filepath}")
    
    results.append({
        'prompt': prompt or 'empty',
        'generated': generated_img
    })

# ===== 7. 重构真实图像测试与SSIM评估 =====
print(f"\n=== 重构真实红外图像测试与SSIM评估 ===")

ssim_scores = []
mse_scores = []
psnr_scores = []

# 从测试集中选择几张图像进行重构
for idx, batch in enumerate(test_dataloader):
    if idx >= 5:  # 只测试前5张
        break
        
    print(f"\n测试真实图像 {idx+1}: {os.path.basename(batch['path'][0])}")
    
    # 获取原始图像tensor
    original_tensor = batch['pixel_values'][0].unsqueeze(0).to(device, dtype=torch.float16)
    
    try:
        # 使用训练VAE编码
        with torch.no_grad():
            encoded_latent = pipe.vae.encode(original_tensor).latent_dist.sample() * 0.18215
        
        # 解码回图像
        with torch.no_grad():
            reconstructed_tensor = pipe.vae.decode(encoded_latent / 0.18215).sample
        
        # 转换为PIL图像
        reconstructed_tensor = (reconstructed_tensor / 2 + 0.5).clamp(0, 1)
        reconstructed_img = transforms.ToPILImage()(reconstructed_tensor[0].cpu().float())
        
        # 读取原始图像
        original_img = Image.open(batch['path'][0]).convert("RGB").resize((image_size, image_size))
        
        # 计算SSIM, MSE, PSNR
        ssim_score = calculate_ssim(original_img, reconstructed_img)
        mse, psnr = calculate_mse_psnr(original_img, reconstructed_img)
        
        ssim_scores.append(ssim_score)
        mse_scores.append(mse)
        psnr_scores.append(psnr)
        
        print(f"  SSIM: {ssim_score:.4f}")
        print(f"  MSE: {mse:.4f}")
        print(f"  PSNR: {psnr:.4f}")
        
        # 保存重构图像
        recon_filename = f"reconstructed_{idx+1}_{os.path.basename(batch['path'][0])}"
        recon_filepath = os.path.join(output_dir, recon_filename)
        reconstructed_img.save(recon_filepath)
        print(f"重构图像已保存: {recon_filepath}")
        
        # 保存原始图像（用于对比）
        orig_filename = f"original_{idx+1}_{os.path.basename(batch['path'][0])}"
        orig_filepath = os.path.join(output_dir, orig_filename)
        original_img.save(orig_filepath)
        print(f"原始图像已保存: {orig_filepath}")
        
    except Exception as e:
        print(f"重构图像 {idx+1} 失败: {e}")

# ===== 8. 批量SSIM评估 =====
print(f"\n=== 重构质量批量评估 ===")
if ssim_scores:
    avg_ssim = sum(ssim_scores) / len(ssim_scores)
    avg_mse = sum(mse_scores) / len(mse_scores)
    avg_psnr = sum(psnr_scores) / len(psnr_scores)
    
    print(f"\n平均重构质量指标:")
    print(f"  平均SSIM: {avg_ssim:.4f}")
    print(f"  平均MSE: {avg_mse:.4f}")
    print(f"  平均PSNR: {avg_psnr:.4f}")
    print(f"  重构图像数量: {len(ssim_scores)}")

# ===== 9. 特征分析 =====
print(f"\n=== 生成图像特征分析 ===")

def analyze_image_features(image_path):
    """分析图像的特征，特别是红外特征"""
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)
    
    # 计算基本统计信息
    r_channel = img_array[:, :, 0]
    g_channel = img_array[:, :, 1] 
    b_channel = img_array[:, :, 2]
    
    stats = {
        'mean_r': r_channel.mean(),
        'mean_g': g_channel.mean(), 
        'mean_b': b_channel.mean(),
        'std_r': r_channel.std(),
        'std_g': g_channel.std(),
        'std_b': b_channel.std(),
        'max_val': img_array.max(),
        'min_val': img_array.min(),
        'channel_corr_rg': np.corrcoef(r_channel.flatten(), g_channel.flatten())[0,1],
        'channel_corr_gb': np.corrcoef(g_channel.flatten(), b_channel.flatten())[0,1],
        'channel_corr_rb': np.corrcoef(r_channel.flatten(), b_channel.flatten())[0,1],
    }
    
    return stats

# 分析生成的红外图像特征
for i, result in enumerate(results):
    img_path = os.path.join(output_dir, f"generated_{i+1}_{result['prompt'].replace(' ', '_')[:20] if result['prompt'] else 'empty'}_512.png")
    if os.path.exists(img_path):
        features = analyze_image_features(img_path)
        print(f"\n生成图像 {i+1} ('{result['prompt']}') 特征:")
        for key, value in features.items():
            if 'corr' in key:
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value:.2f}")

print(f"\n=== 测试完成 ===")
print(f"结果保存在: {output_dir}")