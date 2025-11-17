import sys
import collections
import torch.nn as nn

sys.path.append("..")
import clip.clip as clip
import torch
from torchvision import models, transforms

class CLIPLoss(torch.nn.Module):
    def __init__(self, device, num, affine):
        super(CLIPLoss, self).__init__()

        self.model, clip_preprocess = clip.load(
            'ViT-B/32', device, jit=False)
        self.model.eval()
        self.preprocess = transforms.Compose(
            [clip_preprocess.transforms[-1], clip_preprocess.transforms[0]])  # clip normalisation
        self.device = device
        self.NUM_AUGS = num
        self.mse = nn.MSELoss()
        augemntations = []
        if affine:
            # augemntations.append(transforms.RandomPerspective(
            #     fill=0, p=1.0, distortion_scale=0.5))
            # augemntations.append(transforms.Resize([224, 224]))
            augemntations.append(transforms.Resize([224, 224]))
        augemntations.append(
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)))
        self.augment_trans = transforms.Compose(augemntations)

        self.calc_target = True
        self.counter = 0

    def forward(self, sketches, targets, mode="train"):
        if self.calc_target:
            targets_ = self.preprocess(targets).to(self.device)
            self.targets_features = self.model.encode_image(targets_).detach()
            self.calc_target = False

        if mode == "eval":
            # for regular clip distance, no augmentations
            with torch.no_grad():
                sketches = self.preprocess(sketches).to(self.device)
                sketches_features = self.model.encode_image(sketches)
                return 1. - torch.cosine_similarity(sketches_features, self.targets_features)

        loss_clip = 0
        sketch_augs = []
        img_augs = []
        for n in range(self.NUM_AUGS):
            # augmented_pair = self.augment_trans(torch.cat([sketches, targets]))
            augmented_pair = self.augment_trans(sketches)
            sketch_augs.append(augmented_pair[0].unsqueeze(0))
        sketch_batch = torch.cat(sketch_augs)
        # sketch_utils.plot_batch(img_batch, sketch_batch, self.args, self.counter, use_wandb=False, title="fc_aug{}_iter{}_{}.jpg".format(1, self.counter, mode))
        # if self.counter % 100 == 0:
        # sketch_utils.plot_batch(img_batch, sketch_batch, self.args, self.counter, use_wandb=False, title="aug{}_iter{}_{}.jpg".format(1, self.counter, mode))

        sketch_features = self.model.encode_image(sketch_batch)

        for n in range(self.NUM_AUGS):
            loss_clip += (1. - torch.cosine_similarity(
                sketch_features[n:n+1], self.targets_features, dim=1))
            # loss_clip += self.mse(sketch_features[n:n+1], self.targets_features)
        self.counter += 1
        return loss_clip
        # return 1. - torch.cosine_similarity(sketches_features, self.targets_features)


class CLIPVisualEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.featuremaps = None

        for i in range(12):  # 12 resblocks in VIT visual transformer
            self.clip_model.visual.transformer.resblocks[i].register_forward_hook(
                self.make_hook(i))

    def make_hook(self, name):
        def hook(module, input, output):
            if len(output.shape) == 3:
                self.featuremaps[name] = output.permute(
                    1, 0, 2)  # LND -> NLD bs, smth, 768
            else:
                self.featuremaps[name] = output

        return hook

    def forward(self, x):
        self.featuremaps = collections.OrderedDict()
        fc_features = self.clip_model.encode_image(x).float()
        featuremaps = [self.featuremaps[k] for k in range(12)]

        return fc_features, featuremaps


def l2_layers(xs_conv_features, ys_conv_features, clip_model_name):
    return [torch.square(x_conv - y_conv).mean() for x_conv, y_conv in
            zip(xs_conv_features, ys_conv_features)]


def l1_layers(xs_conv_features, ys_conv_features, clip_model_name):
    return [torch.abs(x_conv - y_conv).mean() for x_conv, y_conv in
            zip(xs_conv_features, ys_conv_features)]


def cos_layers(xs_conv_features, ys_conv_features, clip_model_name):
    if "RN" in clip_model_name:
        return [torch.square(x_conv, y_conv, dim=1).mean() for x_conv, y_conv in
                zip(xs_conv_features, ys_conv_features)]
    return [(1 - torch.cosine_similarity(x_conv, y_conv, dim=1)).mean() for x_conv, y_conv in
            zip(xs_conv_features, ys_conv_features)]


class CLIPConvLoss(torch.nn.Module):
    def __init__(self, device):
        super(CLIPConvLoss, self).__init__()
        self.clip_model_name = "ViT-B/32"
        assert self.clip_model_name in [
            "RN50",
            "RN101",
            "RN50x4",
            "RN50x16",
            "ViT-B/32",
            "ViT-B/16",
        ]

        self.clip_conv_loss_type = "L2"
        self.clip_fc_loss_type = "Cos"  # args.clip_fc_loss_type
        assert self.clip_conv_loss_type in [
            "L2", "Cos", "L1",
        ]
        assert self.clip_fc_loss_type in [
            "L2", "Cos", "L1",
        ]

        self.distance_metrics = \
            {
                "L2": l2_layers,
                "L1": l1_layers,
                "Cos": cos_layers
            }

        self.model, clip_preprocess = clip.load(
            self.clip_model_name, device, jit=False, download_root=".")

        if self.clip_model_name.startswith("ViT"):
            self.visual_encoder = CLIPVisualEncoder(self.model)

        else:
            self.visual_model = self.model.visual
            layers = list(self.model.visual.children())
            init_layers = torch.nn.Sequential(*layers)[:8]
            self.layer1 = layers[8]
            self.layer2 = layers[9]
            self.layer3 = layers[10]
            self.layer4 = layers[11]
            self.att_pool2d = layers[12]


        self.img_size = clip_preprocess.transforms[1].size
        self.model.eval()
        self.target_transform = transforms.Compose([
            transforms.ToTensor(),
        ])  # clip normalisation
        self.normalize_transform = transforms.Compose([
            clip_preprocess.transforms[0],  # Resize
            clip_preprocess.transforms[1],  # CenterCrop
            clip_preprocess.transforms[-1],  # Normalize
        ])

        self.model.eval()
        self.device = device
        # self.num_augs = num

        augemntations = []
        # if affine:
        #     augemntations.append(transforms.RandomPerspective(
        #         fill=0, p=1.0, distortion_scale=0.5))
        #     augemntations.append(transforms.RandomResizedCrop(
        #         224, scale=(0.8, 0.8), ratio=(1.0, 1.0)))
        # else:
        augemntations.append(transforms.Resize([224, 224]))
        augemntations.append(
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)))
        self.augment_trans = transforms.Compose(augemntations)

        self.clip_fc_layer_dims = None  # self.args.clip_fc_layer_dims
        self.clip_conv_layer_dims = None  # self.args.clip_conv_layer_dims
        self.clip_fc_loss_weight = 0.1
        self.counter = 0
        self.clip_conv_layer_weights = "0,0,0,0,0"
        self.clip_conv_layer_weights = [
            float(item) for item in self.clip_conv_layer_weights.split(',')]

    def forward(self, sketch, target, mode="train"):
        """
        Parameters
        ----------
        sketch: Torch Tensor [1, C, H, W]
        target: Torch Tensor [1, C, H, W]
        """
        #         y = self.target_transform(target).to(self.args.device)
        conv_loss_dict = {}
        x = sketch.to(self.device)
        y = target.to(self.device)
        sketch_augs, img_augs = [self.normalize_transform(x)], [
            self.normalize_transform(y)]
        # if mode == "train":
        #     for n in range(self.num_augs):
        #         augmented_pair = self.augment_trans(torch.cat([x, y]))
        #         sketch_augs.append(augmented_pair[0].unsqueeze(0))
        #         img_augs.append(augmented_pair[1].unsqueeze(0))

        xs = torch.cat(sketch_augs, dim=0).to(self.device)
        ys = torch.cat(img_augs, dim=0).to(self.device)

        if self.clip_model_name.startswith("RN"):
            xs_fc_features, xs_conv_features = self.forward_inspection_clip_resnet(
                xs.contiguous())
            ys_fc_features, ys_conv_features = self.forward_inspection_clip_resnet(
                ys.detach())

        else:
            xs_fc_features, xs_conv_features = self.visual_encoder(xs)
            ys_fc_features, ys_conv_features = self.visual_encoder(ys)

        conv_loss = self.distance_metrics[self.clip_conv_loss_type](
            xs_conv_features, ys_conv_features, self.clip_model_name)
        conv_loss_total=0
        for layer, w in enumerate(self.clip_conv_layer_weights):
            if w:
                conv_loss_dict[f"clip_conv_loss_layer{layer}"] = conv_loss[layer] * w
                conv_loss_total = conv_loss_total + conv_loss_dict[f"clip_conv_loss_layer{layer}"]
        if self.clip_fc_loss_weight:
            # fc distance is always cos
            fc_loss = (1 - torch.cosine_similarity(xs_fc_features,
                                                   ys_fc_features, dim=1)).mean()
            conv_loss_dict["fc"] = fc_loss * self.clip_fc_loss_weight
            conv_loss_total += conv_loss_dict["fc"]
        self.counter += 1
        return conv_loss_total

    def forward_inspection_clip_resnet(self, x):
        def stem(m, x):
            x = m.relu1(m.bn1(m.conv1(x)))
            x = m.relu2(m.bn2(m.conv2(x)))
            x = m.relu3(m.bn3(m.conv3(x)))
            x = m.avgpool(x)
            return x

        x = x.type(self.visual_model.conv1.weight.dtype)
        x = stem(self.visual_model, x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        y = self.att_pool2d(x4)
        return y, [x, x1, x2, x3, x4]
