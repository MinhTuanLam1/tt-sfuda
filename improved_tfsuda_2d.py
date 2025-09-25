def parse_args():
    args = {}
    args["source"] = source
    args["target"] = target
    return argparse.Namespace(**args)

def build_strong_augmentation(img):
    """
    Create moderate strong augmentation for a single image tensor.
    Input: img tensor of shape [C, H, W]
    Output: augmented tensor of shape [C, H, W]
    """
    augmentation = []
    # Reduce augmentation intensity for better learning
    augmentation.append(st_transforms.RandomApply([st_transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)], p=0.4))
    augmentation.append(st_transforms.RandomGrayscale(p=0.1))
    # Add geometric augmentations for better robustness
    augmentation.append(st_transforms.RandomApply([st_transforms.RandomRotation(10)], p=0.3))
    strong_aug = st_transforms.Compose(augmentation)
    s_input = strong_aug(img)
    return s_input

def build_pseduo_augmentation(img):
    aug1 = st_transforms.ColorJitter(0.01, 0.01, 0.01, 0.01)
    aug2 = st_transforms.RandomGrayscale(p=1.0)
    aug3 = st_transforms.RandomSolarize(threshold=192.0/255.0,p=1.0)
    aug4 = st_transforms.RandomAutocontrast(p=1.0)
    aug_img1 = aug1(img).unsqueeze(0)
    aug_img2 = aug2(img).unsqueeze(0)
    aug_img3 = aug3(img).unsqueeze(0)
    aug_img4 = aug4(img).unsqueeze(0)
    aug_data = torch.cat([img.unsqueeze(0), aug_img1, aug_img2, aug_img3, aug_img4], dim=0)
    return aug_data

@torch.no_grad()
def update_teacher_model(model_student, model_teacher, keep_rate=0.996):
    student_model_dict = model_student.state_dict()

    new_teacher_dict = OrderedDict()
    for key, value in model_teacher.state_dict().items():
        if key in student_model_dict.keys():
            new_teacher_dict[key] = (
                student_model_dict[key] *
                (1 - keep_rate) + value * keep_rate
            )
        else:
            raise Exception("{} is not found in student model".format(key))
    return new_teacher_dict

def consistency_loss(msrc_feat, tgt_feat):
    req_feat = [0,1,2,3]
    total_loss = 0 
    loss = nn.MSELoss()
    for i in req_feat:
        total_loss = total_loss + loss(tgt_feat[i], msrc_feat[i])
    return total_loss/len(req_feat)


@torch.jit.script
def sigmoid_entropy_loss(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x*torch.log(x + 1e-30) + (1-x)*torch.log(1-x + 1e-30)).mean()

@torch.jit.script
def sigmoid_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x*torch.log(x + 1e-30) + (1-x)*torch.log(1-x + 1e-30))


import torch
import torch.nn.functional as F
import kornia.morphology as kornia_morph

# Giữ nguyên các hàm phụ trợ của bạn: sigmoid_entropy, ent_select

@torch.jit.script
def sigmoid_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x*torch.log(x + 1e-30) + (1-x)*torch.log(1-x + 1e-30))

def ent_select(aug_all_ent):
    aug_req_ent = []
    for i in range(len(aug_all_ent)): 
        if (aug_all_ent[i]).mean().item() > 0.0001: 
            aug_req_ent.append(aug_all_ent[i])
    if not aug_req_ent: # Tránh trường hợp list rỗng
        return aug_all_ent
    return aug_req_ent

import torch
import torch.nn.functional as F
import kornia.morphology as kornia_morph
import torchvision.utils as vutils # Thêm thư viện để debug
import os

# Giữ nguyên các hàm phụ trợ của bạn
@torch.jit.script
def sigmoid_entropy(x: torch.Tensor) -> torch.Tensor:
    return -(x*torch.log(x + 1e-30) + (1-x)*torch.log(1-x + 1e-30))

def ent_select(aug_all_ent):
    aug_req_ent = []
    for i in range(len(aug_all_ent)): 
        if (aug_all_ent[i]).mean().item() > 0.0001: 
            aug_req_ent.append(aug_all_ent[i])
    if not aug_req_ent:
        return aug_all_ent
    return aug_req_ent

# HÀM CẢI TIẾN ĐÃ SỬA LỖI VÀ AN TOÀN HƠN
def uncert_voting_improved(aug_output, 
                          p_threshold=0.5, 
                          ent_percentile=80.0, 
                          uncert_range=(0.3, 0.7),
                          debug_path=None, # Thêm tham số để debug
                          image_idx=0):
    
    # --- 1. Tính toán xác suất và entropy ---
    aug_all_prob = [torch.sigmoid(out) for out in aug_output]
    no_aug_prob = aug_all_prob[0]
    base_pseudo_label = (no_aug_prob >= p_threshold).float()

    no_aug_ent = sigmoid_entropy(no_aug_prob)
    no_aug_ent[torch.isnan(no_aug_ent)] = 0
    
    aug_req_ent = ent_select([sigmoid_entropy(p) for p in aug_all_prob[1:]])
    if not aug_req_ent:
        aug_avg_ent = torch.zeros_like(no_aug_ent)
    else:
        aug_avg_ent = sum(aug_req_ent) / len(aug_req_ent)
    aug_avg_ent[torch.isnan(aug_avg_ent)] = 0

    # --- 2. Tạo bản đồ Entropy có trọng số và dùng ngưỡng động an toàn hơn ---
    ent_weight = 0.75
    no_aug_ent_nor = (no_aug_ent - no_aug_ent.min()) / (no_aug_ent.max() - no_aug_ent.min() + 1e-8)
    aug_avg_ent_nor = (aug_avg_ent - aug_avg_ent.min()) / (aug_avg_ent.max() - aug_avg_ent.min() + 1e-8)
    
    weighted_ent = ent_weight * no_aug_ent_nor + (1 - ent_weight) * aug_avg_ent_nor
    
    # Ngưỡng động dựa trên phân vị
    adaptive_ent_thresh_val = torch.quantile(weighted_ent, ent_percentile / 100.0)
    # Thêm một ngưỡng sàn cố định để tăng độ ổn định
    # 0.3 là một giá trị entropy tương đối (sau chuẩn hóa), có thể tinh chỉnh
    final_ent_thresh = max(adaptive_ent_thresh_val, 0.3) 
    high_entropy_mask = (weighted_ent >= final_ent_thresh).float()

    # --- 3. Chỉ thêm pixel (Logic giống bản gốc nhưng tinh chỉnh hơn) ---
    prob_lower, prob_upper = uncert_range
    uncertain_prob_mask = ((no_aug_prob > prob_lower) & (no_aug_prob < prob_upper)).float()
    
    pixels_to_add = uncertain_prob_mask * high_entropy_mask
    
    # --- 4. Tổng hợp và hậu xử lý an toàn ---
    # Chỉ thêm pixel, không trừ đi
    corrected_pseudo_label = base_pseudo_label + pixels_to_add
    corrected_pseudo_label = torch.clamp(corrected_pseudo_label, 0, 1)

    # Chỉ dùng closing để lấp lỗ hổng, an toàn hơn opening
    kernel = torch.ones(3, 3, device=corrected_pseudo_label.device)
    final_pseudo_label = kornia_morph.closing(corrected_pseudo_label.unsqueeze(0), kernel).squeeze(0)

    # --- 5. (QUAN TRỌNG) Chức năng gỡ lỗi ---
    if debug_path is not None:
        os.makedirs(debug_path, exist_ok=True)
        vutils.save_image(base_pseudo_label, os.path.join(debug_path, f'{image_idx}_0_base_pl.png'), normalize=True)
        vutils.save_image(high_entropy_mask, os.path.join(debug_path, f'{image_idx}_1_high_entropy_mask.png'), normalize=True)
        vutils.save_image(pixels_to_add, os.path.join(debug_path, f'{image_idx}_2_pixels_to_add.png'), normalize=True)
        vutils.save_image(final_pseudo_label, os.path.join(debug_path, f'{image_idx}_3_final_pl.png'), normalize=True)

    return final_pseudo_label.unsqueeze(0).float()

def sfuda_target(config, train_loader, pseduo_model, msrc_model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}
    pseduo_model.eval()
    msrc_model.train()
    pbar = tqdm(total=len(train_loader), disable=True)

    for input, target, path in train_loader:
        aug_input = build_pseduo_augmentation(input.squeeze(0))
        with torch.no_grad():
            aug_output = pseduo_model(aug_input.cuda())
            ps_output = uncert_voting_improved(aug_output.detach())

        optimizer.zero_grad()
        output = msrc_model(aug_input.cuda())
        seg_loss = criterion(output.cuda(), ps_output.repeat(5,1,1,1).cuda())
        ent_loss = sigmoid_entropy_loss(torch.sigmoid(output))
        loss = seg_loss + ent_loss 
        loss.backward()
        optimizer.step()

        iou,dice = iou_score(output, target)
        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()
    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])

def sfuda_task(train_loader, msrc_model, tgt_model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(), 'iou': AverageMeter()}
    msrc_model.eval()
    tgt_model.train()
    pbar = tqdm(total=len(train_loader), disable=True)

    for input, target, _ in train_loader:
        w_input = input.cuda()
        target = target.cuda()
        image_strong_aug = build_strong_augmentation(input.squeeze(0))
        s_input = image_strong_aug.unsqueeze(0).cuda()

        with torch.no_grad():
            w_output, msrc_feat = msrc_model(w_input, mode='const')
            ps_output = torch.sigmoid(w_output).detach().clone()
            ps_output[ps_output>=0.5]=1
            ps_output[ps_output<0.5]=0

        optimizer.zero_grad()
        output, tgt_feat = tgt_model(s_input, mode='const')
        seg_loss = criterion(output, ps_output)
        const_loss = consistency_loss(msrc_feat, tgt_feat)
        loss = seg_loss + const_loss
        loss.backward()
        optimizer.step()

        iou, dice = iou_score(output, target)
        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
            ])
        pbar.set_postfix(postfix)
        pbar.update(1)

        new_msrc_dict = update_teacher_model(tgt_model, msrc_model, keep_rate=0.99)
        msrc_model.load_state_dict(new_msrc_dict)
        
    pbar.close()
    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])


def validate(val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter()}

    model.eval()
    with torch.no_grad():
        pbar = tqdm(total=len(val_loader), disable=True)
        for input, target, meta in val_loader:
            input = input.cuda()
            target = target.cuda()

            output = model(input)
            loss = criterion(output, target)
            iou,dice = iou_score(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()
    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg)])

def main_improved():
    args = parse_args()

    config_file = "config_" + args.target
    with open('models/%s/%s.yml' % (args.source, config_file), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    train_img_ids = glob(os.path.join('inputs', args.target, 'train','images', '*' + config['img_ext']))
    train_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in train_img_ids]

    val_img_ids = glob(os.path.join('inputs', args.target, 'test','images', '*' + config['img_ext']))
    val_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in val_img_ids]

    train_transform = Compose([
        RandomRotate90(),
        transforms.Flip(),
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join('inputs', args.target, 'train','images'),
        mask_dir=os.path.join('inputs', args.target, 'train','masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=train_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', args.target,'test', 'images'),
        mask_dir=os.path.join('inputs', args.target,'test', 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    print("Creating model %s...!!!" % config['arch'])
    print("Loading source trained model...!!!")
    msrc_model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])

    msrc_model.load_state_dict(torch.load('models/%s/model.pth'%config['name']))
    msrc_model.cuda()
    msrc_model.train()
    print("Sucessfully loaded source trained model...!!!")

    tgt_model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])
    tgt_model.cuda()
    tgt_model.train()

    src_params = filter(lambda p: p.requires_grad, msrc_model.parameters())
    src_optimizer = optim.Adam(src_params, lr=config['lr'], weight_decay=config['weight_decay'])

    tgt_params = filter(lambda p: p.requires_grad, tgt_model.parameters())
    tgt_optimizer = optim.Adam(tgt_params, lr=config['lr'], weight_decay=config['weight_decay'])

    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
    
    pseudo_model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])
    pretrained_dict = msrc_model.state_dict()
    pseudo_model.load_state_dict(pretrained_dict)
    pseudo_model.cuda()
    pseudo_model.eval()

    criterion = losses.__dict__[config['loss']]().cuda()
    
    print("")
    print("Performing source only model evaluation...!!!")
    val_log = validate(val_loader, msrc_model, criterion)
    print('Source_only dice: %.4f' % (val_log['dice']))
    source_trained_only_dice =  val_log['dice']
    print("")
    print("Target specific adaptation (Stage 1)...!!!")
    for epoch in range(config['stage1']):
        train_log = sfuda_target(config, train_loader, pseudo_model, msrc_model, criterion, src_optimizer)
        print('Epoch %d - train_loss %.4f - train_iou %.4f' % (epoch+1, train_log['loss'], train_log['iou']))

    msrc_model.eval()
    pretrained_dict = msrc_model.state_dict()
    tgt_model.load_state_dict(pretrained_dict)
    tgt_model.cuda()
    tgt_model.train()

    print("")
    print("Task specific adaptation (Stage 2 - IMPROVED)...!!!")
    for epoch in range(config['stage2']):
        train_log = sfuda_task(train_loader, msrc_model, tgt_model, criterion, tgt_optimizer)
        print('Epoch %d - total_loss %.4f - seg_loss %.4f - const_loss %.4f - contrastive_loss %.4f - iou %.4f' % (
            epoch+1, train_log['loss'], train_log['seg_loss'], 
            train_log['const_loss'], train_log['contrastive_loss'], train_log['iou']
        ))
    
    print("")
    print("Performing adapted target model evaluation...!!!")
    val_log = validate(val_loader, tgt_model, criterion)
    print('Adapted target model dice: %.4f' % (val_log['dice']))
    
    print("")
    print("=== PERFORMANCE COMPARISON ===")
    print('Source-only dice: %.4f' % source_trained_only_dice)
    print('Improved TT-SFUDA dice: %.4f' % val_log['dice'])
    print('Improvement: %.4f' % (val_log['dice'] - source_trained_only_dice))
    
    return val_log['dice'], source_trained_only_dice

# Định nghĩa source và target
source = "chase_unet"  # Có thể thay đổi theo nhu cầu
target = "hrf"         # Có thể thay đổi theo nhu cầu
# === REGULARIZATION TECHNIQUES ===

class DropoutRegularization(nn.Module):
    """
    Adaptive dropout cho regularization
    """
    def __init__(self, dropout_rate=0.1):
        super().__init__()
        self.dropout = nn.Dropout2d(dropout_rate)
        
    def forward(self, x, training=True):
        if training:
            return self.dropout(x)
        return x

class SpectralNormalization:
    """
    Spectral normalization để ổn định training
    """
    @staticmethod
    def apply_spectral_norm(model):
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.utils.spectral_norm(module)
        return model

def gradient_penalty(model, real_data, fake_data, device):
    """
    Gradient penalty cho regularization
    """
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1).to(device)
    
    interpolated = alpha * real_data + (1 - alpha) * fake_data
    interpolated.requires_grad_(True)
    
    output = model(interpolated)
    
    gradients = torch.autograd.grad(
        outputs=output,
        inputs=interpolated,
        grad_outputs=torch.ones_like(output),
        create_graph=True,
        retain_graph=True
    )[0]
    
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = ((gradient_norm - 1) ** 2).mean()
    
    return penalty

# === PERFORMANCE MONITORING ===

class PerformanceMonitor:
    """
    Monitor training performance và detect issues
    """
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.loss_history = []
        self.gradient_norms = []
        
    def update(self, loss, model):
        self.loss_history.append(loss)
        
        # Tính gradient norm
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        self.gradient_norms.append(total_norm)
        
        # Giữ chỉ window_size samples gần nhất
        if len(self.loss_history) > self.window_size:
            self.loss_history.pop(0)
            self.gradient_norms.pop(0)
    
    def check_convergence(self):
        """
        Kiểm tra convergence issues
        """
        if len(self.loss_history) < self.window_size:
            return {"status": "insufficient_data"}
        
        recent_losses = self.loss_history[-5:]
        loss_variance = np.var(recent_losses)
        loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
        
        avg_gradient_norm = np.mean(self.gradient_norms[-5:])
        
        issues = []
        if loss_variance < 1e-6:
            issues.append("loss_plateau")
        if abs(loss_trend) < 1e-5:
            issues.append("no_improvement")
        if avg_gradient_norm < 1e-6:
            issues.append("vanishing_gradients")
        if avg_gradient_norm > 10:
            issues.append("exploding_gradients")
            
        return {
            "status": "issues_detected" if issues else "healthy",
            "issues": issues,
            "loss_variance": loss_variance,
            "loss_trend": loss_trend,
            "avg_gradient_norm": avg_gradient_norm
        }

def soft_pseudo_label_generation(teacher_output, confidence_threshold=0.6, temperature=2.0):
    """
    Generate soft pseudo-labels with confidence filtering and temperature scaling
    FOR BINARY SEGMENTATION
    
    Args:
        teacher_output: Raw teacher model output (logits) - shape [B, 1, H, W]
        confidence_threshold: Minimum confidence for pseudo-label acceptance (lowered for binary)
        temperature: Temperature for sigmoid scaling (higher = softer)
    
    Returns:
        soft_labels: Temperature-scaled soft labels [B, 1, H, W]
        confidence_mask: Binary mask indicating high-confidence pixels
    """
    # Apply temperature scaling for softer predictions
    scaled_logits = teacher_output / temperature
    soft_labels = torch.sigmoid(scaled_logits)  # Use sigmoid for binary segmentation
    
    # Calculate confidence based on distance from 0.5 (uncertainty)
    confidence = torch.abs(soft_labels - 0.5) * 2  # Convert to [0,1] range
    confidence_mask = (confidence > confidence_threshold).float()
    
    return soft_labels, confidence_mask

class PixelContrastiveLoss(nn.Module):
    """
    Pixel-level contrastive learning using UNet feature maps
    """
    def __init__(self, temperature=0.1, num_negatives=256):
        super().__init__()
        self.temperature = temperature
        self.num_negatives = num_negatives
        
    def forward(self, feat_weak_list, feat_strong_list, pseudo_labels):
        """
        Args:
            feat_weak_list: List of feature maps from weak augmentation [x1_0, x2_0, x3_0, x4_0]
            feat_strong_list: List of feature maps from strong augmentation [x1_0, x2_0, x3_0, x4_0]
            pseudo_labels: Pseudo labels [B, H, W] or [B, 1, H, W]
        """
        if len(feat_weak_list) != len(feat_strong_list):
            return torch.tensor(0.0, device=pseudo_labels.device, requires_grad=True)
        
        # Ensure pseudo_labels has 4 dimensions [B, 1, H, W]
        if pseudo_labels.dim() == 3:
            pseudo_labels = pseudo_labels.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
        
        total_loss = 0
        num_levels = 0
        
        # Use features from different resolution levels
        for feat_weak, feat_strong in zip(feat_weak_list, feat_strong_list):
            B, C, H, W = feat_weak.shape
            
            # Resize pseudo labels to match feature map size
            labels_resized = F.interpolate(pseudo_labels.float(), size=(H, W), mode='nearest')
            
            # Flatten spatial dimensions
            feat_weak_flat = feat_weak.view(B, C, -1).permute(0, 2, 1)  # [B, HW, C]
            feat_strong_flat = feat_strong.view(B, C, -1).permute(0, 2, 1)  # [B, HW, C]
            labels_flat = labels_resized.view(B, -1)  # [B, HW]
            
            # Normalize features
            feat_weak_flat = F.normalize(feat_weak_flat, dim=2)
            feat_strong_flat = F.normalize(feat_strong_flat, dim=2)
            
            level_loss = 0
            valid_batches = 0
            
            for b in range(B):
                # Get positive and negative pixel indices
                pos_mask = (labels_flat[b] == 1).nonzero(as_tuple=False).squeeze(1)
                neg_mask = (labels_flat[b] == 0).nonzero(as_tuple=False).squeeze(1)
                
                if len(pos_mask) == 0 or len(neg_mask) == 0:
                    continue
                
                # Sample a subset to avoid memory issues
                max_pos = min(len(pos_mask), 64)  # Limit positive samples
                max_neg = min(len(neg_mask), self.num_negatives)
                
                if len(pos_mask) > max_pos:
                    pos_indices = torch.randperm(len(pos_mask))[:max_pos]
                    pos_mask = pos_mask[pos_indices]
                
                if len(neg_mask) > max_neg:
                    neg_indices = torch.randperm(len(neg_mask))[:max_neg]
                    neg_mask = neg_mask[neg_indices]
                
                # Get features
                pos_feat_weak = feat_weak_flat[b][pos_mask]  # [N_pos, C]
                pos_feat_strong = feat_strong_flat[b][pos_mask]  # [N_pos, C]
                neg_feat_weak = feat_weak_flat[b][neg_mask]  # [N_neg, C]
                
                # Positive similarities (same pixel, different augmentation)
                pos_sim = torch.sum(pos_feat_weak * pos_feat_strong, dim=1) / self.temperature
                
                # Negative similarities (different pixels)
                neg_sim = torch.mm(pos_feat_strong, neg_feat_weak.t()) / self.temperature
                
                # Contrastive loss
                logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
                labels_contrastive = torch.zeros(len(pos_mask), dtype=torch.long, device=logits.device)
                
                loss = F.cross_entropy(logits, labels_contrastive)
                level_loss += loss
                valid_batches += 1
            
            if valid_batches > 0:
                total_loss += level_loss / valid_batches
                num_levels += 1
        
        return total_loss / max(num_levels, 1)

def adaptive_loss_weights(epoch, total_epochs, seg_loss, const_loss, contrastive_loss):
    """
    Dynamically balance loss weights based on training progress and loss magnitudes
    """
    # Progressive weight scheduling
    progress = epoch / total_epochs
    
    # Segmentation loss: Start high, gradually decrease
    w_seg = 1.0 - 0.3 * progress
    
    # Consistency loss: Gradually increase
    w_const = 0.5 + 0.5 * progress
    
    # Contrastive loss: Peak in middle of training
    w_contrastive = 0.5 * (1 - abs(2 * progress - 1))
    
    # Adaptive balancing based on loss magnitudes
    losses = torch.tensor([seg_loss, const_loss, contrastive_loss])
    loss_ratios = losses / (losses.mean() + 1e-8)
    
    # Reduce weight for dominant losses
    adaptive_factors = torch.clamp(2.0 - loss_ratios, 0.5, 2.0)
    
    w_seg *= adaptive_factors[0].item()
    w_const *= adaptive_factors[1].item()
    w_contrastive *= adaptive_factors[2].item()
    
    return w_seg, w_const, w_contrastive

def update_teacher_model_adaptive(teacher_model, student_model, keep_rate=0.99, momentum_schedule=None, epoch=0):
    """
    Adaptive teacher model update with scheduled momentum and keep_rate
    """
    if momentum_schedule is not None:
        # Use scheduled momentum if provided
        if epoch < len(momentum_schedule):
            current_keep_rate = momentum_schedule[epoch]
        else:
            current_keep_rate = momentum_schedule[-1]
    else:
        # Default adaptive scheduling
        # Start with higher momentum (lower keep_rate) for faster adaptation
        # Gradually increase keep_rate for stability
        if epoch < 10:
            current_keep_rate = 0.95 + 0.004 * epoch  # 0.95 -> 0.986
        else:
            current_keep_rate = keep_rate
    
    # EMA update using state_dict for compatibility
    student_model_dict = student_model.state_dict()
    new_teacher_dict = OrderedDict()
    
    for key, value in teacher_model.state_dict().items():
        if key in student_model_dict.keys():
            new_teacher_dict[key] = (
                student_model_dict[key] * (1 - current_keep_rate) + 
                value * current_keep_rate
            )
        else:
            new_teacher_dict[key] = value
    
    teacher_model.load_state_dict(new_teacher_dict)
    return current_keep_rate

def sfuda_task_improved_1(train_loader, msrc_model, tgt_model, criterion, optimizer, 
                       epoch=0, update_freq=3, use_contrastive=True, total_epochs=50):
    """
    Improved Stage 2: Enhanced teacher-student learning with multiple improvements
    """
    avg_meters = {'loss': AverageMeter(), 'seg_loss': AverageMeter(), 
                  'const_loss': AverageMeter(), 'contrastive_loss': AverageMeter(),
                  'iou': AverageMeter(), 'dice': AverageMeter()}

    # Initialize contrastive loss if enabled
    contrastive_criterion = PixelContrastiveLoss() if use_contrastive else None
    
    # Teacher model in eval mode, student in train mode
    msrc_model.eval()
    tgt_model.train()

    pbar = tqdm(total=len(train_loader), disable=True)
    
    for input, target, meta in train_loader:
        input = input.cuda()
        target = target.cuda()

        # Strong augmentation for student
        strong_aug_input = build_strong_augmentation(input[0]).unsqueeze(0)
        
        # Teacher prediction with weak augmentation (original input)
        with torch.no_grad():
            teacher_output, teacher_features = msrc_model(input, mode='const')
            
        # Generate soft pseudo-labels with confidence filtering
        soft_labels, confidence_mask = soft_pseudo_label_generation(
            teacher_output, confidence_threshold=0.7, temperature=2.0
        )
        
        # Student prediction with strong augmentation
        student_output, student_features = tgt_model(strong_aug_input, mode='const')
        
        # Segmentation loss (only on high-confidence pixels)
        seg_loss = criterion(student_output, soft_labels)
        seg_loss = (seg_loss * confidence_mask.unsqueeze(1)).mean()
        
        # Consistency loss between teacher and student features
        const_loss = consistency_loss(teacher_features, student_features)
        
        # Contrastive loss for better feature representation
        contrastive_loss_val = 0
        if use_contrastive and contrastive_criterion is not None:
            # Use hard pseudo-labels for contrastive learning - fix for binary segmentation
            hard_pseudo_labels = (soft_labels > 0.5).float().squeeze(1)  # Binary threshold
            contrastive_loss_val = contrastive_criterion(
                teacher_features, student_features, hard_pseudo_labels
            )
        
        # Simplified loss weighting for better stability
        w_seg = 1.0  # Keep segmentation loss as primary
        w_const = 0.1  # Reduce consistency loss weight
        w_contrastive = 0.05  # Small contrastive weight
        
        # Total loss with fixed weights for stability
        total_loss = (w_seg * seg_loss + 
                     w_const * const_loss + 
                     w_contrastive * contrastive_loss_val)

        # Backward pass with gradient clipping
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(tgt_model.parameters(), max_norm=1.0)
        optimizer.step()

        # Calculate metrics - IMPORTANT: Use ground truth target, not soft labels!
        iou, dice = iou_score(student_output, target)
        
        # Debug: Monitor confidence mask coverage
        confidence_coverage = confidence_mask.mean().item()
        
        # Debug: Check if pseudo-labels are reasonable
        pseudo_label_mean = soft_labels.mean().item()
        teacher_target_iou, _ = iou_score(teacher_output, target)  # Teacher performance on target

        # Update meters
        avg_meters['loss'].update(total_loss.item(), input.size(0))
        avg_meters['seg_loss'].update(seg_loss.item(), input.size(0))
        avg_meters['const_loss'].update(const_loss.item(), input.size(0))
        avg_meters['contrastive_loss'].update(
            contrastive_loss_val.item() if isinstance(contrastive_loss_val, torch.Tensor) else contrastive_loss_val, 
            input.size(0)
        )
        avg_meters['iou'].update(iou, input.size(0))
        avg_meters['dice'].update(dice, input.size(0))
        
        # Add debug meters
        if 'confidence_coverage' not in avg_meters:
            avg_meters['confidence_coverage'] = AverageMeter()
            avg_meters['pseudo_label_mean'] = AverageMeter()
            avg_meters['teacher_iou'] = AverageMeter()
        
        avg_meters['confidence_coverage'].update(confidence_coverage, input.size(0))
        avg_meters['pseudo_label_mean'].update(pseudo_label_mean, input.size(0))
        avg_meters['teacher_iou'].update(teacher_target_iou, input.size(0))

        # Scheduled teacher model update (less frequent for stability)
        if (pbar.n + 1) % update_freq == 0:
            current_keep_rate = update_teacher_model_adaptive(
                msrc_model, tgt_model, keep_rate=0.99, epoch=epoch
            )

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('seg', avg_meters['seg_loss'].avg),
            ('const', avg_meters['const_loss'].avg),
            ('contr', avg_meters['contrastive_loss'].avg),
            ('iou', avg_meters['iou'].avg),
            ('dice', avg_meters['dice'].avg),
            ('conf_cov', avg_meters['confidence_coverage'].avg),
            ('t_iou', avg_meters['teacher_iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    
    pbar.close()

    return OrderedDict([
        ('loss', avg_meters['loss'].avg),
        ('seg_loss', avg_meters['seg_loss'].avg),
        ('const_loss', avg_meters['const_loss'].avg),
        ('contrastive_loss', avg_meters['contrastive_loss'].avg),
        ('iou', avg_meters['iou'].avg),
        ('dice', avg_meters['dice'].avg),
        ('confidence_coverage', avg_meters['confidence_coverage'].avg),
        ('teacher_iou', avg_meters['teacher_iou'].avg),
        ('pseudo_label_mean', avg_meters['pseudo_label_mean'].avg)
    ])

def validate(val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter()}

    model.eval()
    with torch.no_grad():
        pbar = tqdm(total=len(val_loader), disable=True)
        for input, target, meta in val_loader:
            input = input.cuda()
            target = target.cuda()

            output = model(input)
            loss = criterion(output, target)
            iou,dice = iou_score(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()
    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg)])

def main_improved():
    args = parse_args()

    config_file = "config_" + args.target
    with open('models/%s/%s.yml' % (args.source, config_file), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    train_img_ids = glob(os.path.join('inputs', args.target, 'train','images', '*' + config['img_ext']))
    train_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in train_img_ids]

    val_img_ids = glob(os.path.join('inputs', args.target, 'test','images', '*' + config['img_ext']))
    val_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in val_img_ids]

    train_transform = Compose([
        RandomRotate90(),
        transforms.Flip(),
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join('inputs', args.target, 'train','images'),
        mask_dir=os.path.join('inputs', args.target, 'train','masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=train_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', args.target,'test', 'images'),
        mask_dir=os.path.join('inputs', args.target,'test', 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    print("Creating model %s...!!!" % config['arch'])
    print("Loading source trained model...!!!")
    msrc_model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])

    msrc_model.load_state_dict(torch.load('models/%s/model.pth'%config['name']))
    msrc_model.cuda()
    msrc_model.train()
    print("Sucessfully loaded source trained model...!!!")

    tgt_model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])
    tgt_model.cuda()
    tgt_model.train()

    src_params = filter(lambda p: p.requires_grad, msrc_model.parameters())
    src_optimizer = optim.Adam(src_params, lr=config['lr'], weight_decay=config['weight_decay'])

    tgt_params = filter(lambda p: p.requires_grad, tgt_model.parameters())
    tgt_optimizer = optim.Adam(tgt_params, lr=config['lr'], weight_decay=config['weight_decay'])

    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
    
    pseudo_model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])
    pretrained_dict = msrc_model.state_dict()
    pseudo_model.load_state_dict(pretrained_dict)
    pseudo_model.cuda()
    pseudo_model.eval()

    criterion = losses.__dict__[config['loss']]().cuda()
    
    print("")
    print("Performing source only model evaluation...!!!")
    val_log = validate(val_loader, msrc_model, criterion)
    print('Source_only dice: %.4f' % (val_log['dice']))
    source_trained_only_dice =  val_log['dice']
    print("")
    print("Target specific adaptation (Stage 1)...!!!")
    for epoch in range(config['stage1']):
        train_log = sfuda_target(config, train_loader, pseudo_model, msrc_model, criterion, src_optimizer)
        print('Epoch %d - train_loss %.4f - train_iou %.4f' % (epoch+1, train_log['loss'], train_log['iou']))

    msrc_model.eval()
    pretrained_dict = msrc_model.state_dict()
    tgt_model.load_state_dict(pretrained_dict)
    tgt_model.cuda()
    tgt_model.train()

    print("")
    print("Task specific adaptation (Stage 2 - IMPROVED)...!!!")
    for epoch in range(config['stage2']):
        train_log = sfuda_task(train_loader, msrc_model, tgt_model, criterion, tgt_optimizer)
        print('Epoch %d - total_loss %.4f - seg_loss %.4f - const_loss %.4f - contrastive_loss %.4f - iou %.4f' % (
            epoch+1, train_log['loss'], train_log['seg_loss'], 
            train_log['const_loss'], train_log['contrastive_loss'], train_log['iou']
        ))
    
    print("")
    print("Performing adapted target model evaluation...!!!")
    val_log = validate(val_loader, tgt_model, criterion)
    print('Adapted target model dice: %.4f' % (val_log['dice']))
    
    print("")
    print("=== PERFORMANCE COMPARISON ===")
    print('Source-only dice: %.4f' % source_trained_only_dice)
    print('Improved TT-SFUDA dice: %.4f' % val_log['dice'])
    print('Improvement: %.4f' % (val_log['dice'] - source_trained_only_dice))
    
    return val_log['dice'], source_trained_only_dice

# Định nghĩa source và target
source = "chase_unet"  # Có thể thay đổi theo nhu cầu
target = "hrf"         # Có thể thay đổi theo nhu cầu

# === REGULARIZATION TECHNIQUES ===

class DropoutRegularization(nn.Module):
    """
    Adaptive dropout cho regularization
    """
    def __init__(self, dropout_rate=0.1):
        super().__init__()
        self.dropout = nn.Dropout2d(dropout_rate)
        
    def forward(self, x, training=True):
        if training:
            return self.dropout(x)
        return x

class SpectralNormalization:
    """
    Spectral normalization để ổn định training
    """
    @staticmethod
    def apply_spectral_norm(model):
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.utils.spectral_norm(module)
        return model

def gradient_penalty(model, real_data, fake_data, device):
    """
    Gradient penalty cho regularization
    """
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1).to(device)
    
    interpolated = alpha * real_data + (1 - alpha) * fake_data
    interpolated.requires_grad_(True)
    
    output = model(interpolated)
    
    gradients = torch.autograd.grad(
        outputs=output,
        inputs=interpolated,
        grad_outputs=torch.ones_like(output),
        create_graph=True,
        retain_graph=True
    )[0]
    
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = ((gradient_norm - 1) ** 2).mean()
    
    return penalty

# === PERFORMANCE MONITORING ===

class PerformanceMonitor:
    """
    Monitor training performance và detect issues
    """
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.loss_history = []
        self.gradient_norms = []
        
    def update(self, loss, model):
        self.loss_history.append(loss)
        
        # Tính gradient norm
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        self.gradient_norms.append(total_norm)
        
        # Giữ chỉ window_size samples gần nhất
        if len(self.loss_history) > self.window_size:
            self.loss_history.pop(0)
            self.gradient_norms.pop(0)
    
    def check_convergence(self):
        """
        Kiểm tra convergence issues
        """
        if len(self.loss_history) < self.window_size:
            return {"status": "insufficient_data"}
        
        recent_losses = self.loss_history[-5:]
        loss_variance = np.var(recent_losses)
        loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
        
        avg_gradient_norm = np.mean(self.gradient_norms[-5:])
        
        issues = []
        if loss_variance < 1e-6:
            issues.append("loss_plateau")
        if abs(loss_trend) < 1e-5:
            issues.append("no_improvement")
        if avg_gradient_norm < 1e-6:
            issues.append("vanishing_gradients")
        if avg_gradient_norm > 10:
            issues.append("exploding_gradients")
            
        return {
            "status": "issues_detected" if issues else "healthy",
            "issues": issues,
            "loss_variance": loss_variance,
            "loss_trend": loss_trend,
            "avg_gradient_norm": avg_gradient_norm
        }

def main_improved_stage2():
    """
    Main function sử dụng Stage 2 cải tiến với performance monitoring
    """
    args = parse_args()

    config_file = "config_" + args.target
    with open('models/%s/%s.yml' % (args.source, config_file), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Data loading (giống như main_improved)
    train_img_ids = glob(os.path.join('inputs', args.target, 'train','images', '*' + config['img_ext']))
    train_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in train_img_ids]

    val_img_ids = glob(os.path.join('inputs', args.target, 'test','images', '*' + config['img_ext']))
    val_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in val_img_ids]

    train_transform = Compose([
        RandomRotate90(),
        transforms.Flip(),
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join('inputs', args.target, 'train','images'),
        mask_dir=os.path.join('inputs', args.target, 'train','masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=train_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', args.target,'test', 'images'),
        mask_dir=os.path.join('inputs', args.target,'test', 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    # Model setup
    print("Creating model %s...!!!" % config['arch'])
    print("Loading source trained model...!!!")
    msrc_model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])

    msrc_model.load_state_dict(torch.load('models/%s/model.pth'%config['name']))
    msrc_model.cuda()
    msrc_model.train()
    print("Successfully loaded source trained model...!!!")

    tgt_model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])
    tgt_model.cuda()
    tgt_model.train()

    src_params = filter(lambda p: p.requires_grad, msrc_model.parameters())
    src_optimizer = optim.Adam(src_params, lr=config['lr'], weight_decay=config['weight_decay'])

    tgt_params = filter(lambda p: p.requires_grad, tgt_model.parameters())
    tgt_optimizer = optim.Adam(tgt_params, lr=config['lr'], weight_decay=config['weight_decay'])

    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
    
    pseudo_model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])
    pretrained_dict = msrc_model.state_dict()
    pseudo_model.load_state_dict(pretrained_dict)
    pseudo_model.cuda()
    pseudo_model.eval()

    criterion = losses.__dict__[config['loss']]().cuda()
    
    # === BASELINE EVALUATION ===
    print("")
    print("Performing source only model evaluation...!!!")
    val_log = validate(val_loader, msrc_model, criterion)
    print('Source_only dice: %.4f' % (val_log['dice']))
    source_trained_only_dice = val_log['dice']
    
    # === STAGE 1: TARGET SPECIFIC ADAPTATION ===
    print("")
    print("Target specific adaptation...!!!")
    for epoch in range(config['stage1']):
        train_log = sfuda_target(config, train_loader, pseudo_model, msrc_model, criterion, src_optimizer)
        print('train_loss %.4f - train_iou %.4f' % (train_log['loss'], train_log['iou']))

    msrc_model.eval()
    pretrained_dict = msrc_model.state_dict()
    tgt_model.load_state_dict(pretrained_dict)
    tgt_model.cuda()
    tgt_model.train()

    # === STAGE 2: IMPROVED TASK SPECIFIC ADAPTATION ===
    print("")
    print("=== IMPROVED Task Specific Adaptation ===")
    print("Features: Soft pseudo-labels, Contrastive learning, Adaptive loss balancing")
    
    # Performance tracking
    stage2_metrics = {
        'epoch': [],
        'total_loss': [],
        'seg_loss': [],
        'const_loss': [],
        'contrastive_loss': [],
        'iou': [],
        'val_dice': []  # Track validation performance
    }
    
    best_val_dice = 0.0
    patience = 5
    patience_counter = 0
    
    for epoch in range(config['stage2']):
        print(f"Epoch {epoch+1}/{config['stage2']}")
        
        # Sử dụng improved Stage 2
        train_log = sfuda_task_improved_1(
            train_loader, msrc_model, tgt_model, criterion, tgt_optimizer, 
            epoch=epoch, update_freq=3, use_contrastive=True
        )
        
        # Log metrics
        stage2_metrics['epoch'].append(epoch + 1)
        stage2_metrics['total_loss'].append(train_log['loss'])
        stage2_metrics['seg_loss'].append(train_log['seg_loss'])
        stage2_metrics['const_loss'].append(train_log['const_loss'])
        stage2_metrics['contrastive_loss'].append(train_log['contrastive_loss'])
        stage2_metrics['iou'].append(train_log['iou'])
        
        print('Total Loss: %.4f | Seg Loss: %.4f | Const Loss: %.4f | Contrastive: %.4f | IoU: %.4f' % 
              (train_log['loss'], train_log['seg_loss'], train_log['const_loss'], 
               train_log['contrastive_loss'], train_log['iou']))
        
        # Debug metrics logging
        if 'confidence_coverage' in train_log:
            print('  -> Debug: Conf Coverage: %.3f | Teacher IoU: %.4f | Pseudo Mean: %.3f' % 
                  (train_log['confidence_coverage'], train_log['teacher_iou'], train_log['pseudo_label_mean']))
        
        # Early warning for problematic training
        if train_log['confidence_coverage'] < 0.1:
            print("  ⚠️  WARNING: Very low confidence coverage - pseudo-labels might be too strict!")
        if train_log['teacher_iou'] < 0.3:
            print("  ⚠️  WARNING: Teacher model performing poorly on target domain!")
        if abs(train_log['pseudo_label_mean'] - 0.5) > 0.4:
            print("  ⚠️  WARNING: Pseudo-labels heavily biased - check class imbalance!")
        
        # Validation mỗi 3 epochs để theo dõi overfitting
        if (epoch + 1) % 3 == 0:
            val_log = validate(val_loader, tgt_model, criterion)
            val_dice = val_log['dice']
            stage2_metrics['val_dice'].append(val_dice)
            print(f'  -> Validation Dice: %.4f' % val_dice)
            
            # Early stopping based on validation performance
            if val_dice > best_val_dice:
                best_val_dice = val_dice
                patience_counter = 0
                print(f"  ✅ New best validation Dice: {best_val_dice:.4f}")
            else:
                patience_counter += 1
                print(f"  📉 No improvement for {patience_counter} validation checks")
                
                if patience_counter >= patience:
                    print(f"  🛑 Early stopping triggered! Best validation Dice: {best_val_dice:.4f}")
                    break
    
    # === FINAL EVALUATION ===
    print("")
    print("=== FINAL EVALUATION ===")
    val_log = validate(val_loader, tgt_model, criterion)
    final_dice = val_log['dice']
    print('Improved target model dice: %.4f' % final_dice)
    
    # === PERFORMANCE COMPARISON ===
    print("")
    print("=== PERFORMANCE COMPARISON ===")
    print(f'Source-only Dice: {source_trained_only_dice:.4f}')
    print(f'Improved TT-SFUDA Dice: {final_dice:.4f}')
    improvement = final_dice - source_trained_only_dice
    print(f'Improvement: {improvement:.4f} ({improvement/source_trained_only_dice*100:.2f}%)')
    
    # === TRAINING STATISTICS ===
    print("")
    print("=== STAGE 2 TRAINING STATISTICS ===")
    print(f'Average Total Loss: {sum(stage2_metrics["total_loss"])/len(stage2_metrics["total_loss"]):.4f}')
    print(f'Average Segmentation Loss: {sum(stage2_metrics["seg_loss"])/len(stage2_metrics["seg_loss"]):.4f}')
    print(f'Average Consistency Loss: {sum(stage2_metrics["const_loss"])/len(stage2_metrics["const_loss"]):.4f}')
    print(f'Average Contrastive Loss: {sum(stage2_metrics["contrastive_loss"])/len(stage2_metrics["contrastive_loss"]):.4f}')
    print(f'Final IoU: {stage2_metrics["iou"][-1]:.4f}')
    
    return {
        'source_dice': source_trained_only_dice,
        'final_dice': final_dice,
        'improvement': improvement,
        'stage2_metrics': stage2_metrics
    }



if __name__ == '__main__':
    # Chạy improved version
    results = main_improved_stage2()
    print("\n=== EXPERIMENT COMPLETED ===")
    print(f"Final improvement: {results['improvement']:.4f}")