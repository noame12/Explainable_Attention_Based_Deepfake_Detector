
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
# import torchvision.transforms as transforms
# import torch.nn as nn
from torch.nn import AvgPool2d
import yaml
from PIL import Image
import os
import csv
from albumentations import Compose,PadIfNeeded
from baselines.EfficientViT.transforms.albu import IsotropicResize




# rule 5 from paper
def avg_heads(cam, grad):
    cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
    cam = grad * cam
    cam = cam.clamp(min=0).mean(dim=0)
    return cam


# rule 6 from paper
def apply_self_attention_rules(R_ss, cam_ss):
    R_ss_addition = torch.matmul(cam_ss, R_ss)
    return R_ss_addition


def generate_relevance(model, input):
    output = model(input, register_hook=True)
    model.zero_grad()
    output.backward(retain_graph=True)

    num_tokens = model.transformer.blocks[0].attn.get_attention_map().shape[-1]
    R = torch.eye(num_tokens, num_tokens).cuda()
    for blk in model.transformer.blocks:
        grad = blk.attn.get_attn_gradients()
        # g_view = grad.cpu().numpy()
        cam = blk.attn.get_attention_map()
        cam = avg_heads(cam, grad)
        R += apply_self_attention_rules(R.cuda(), cam.cuda())
    return R[0, 1:]


from baselines.EfficientViT.evit_model10 import EfficientViT


# normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

def create_base_transform(size): #fixme: added from test evit
    return Compose([
        IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
    ])

# create heatmap from mask on image
def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam



BASE_DIR = '../deep_fakes_explain/'
DATA_DIR = os.path.join(BASE_DIR, "dataset")
TEST_DIR = os.path.join(DATA_DIR, "validation_set")
MODELS_PATH = os.path.join(BASE_DIR, "models1")
EXAMPLES_DIR = 'examples'

# Initialize the Efficientnet_ViT Deepfake Detector pretrained model
config = 'baselines/EfficientViT/explained_architecture1.yaml'
with open(config, 'r') as ymlfile:
    config = yaml.safe_load(ymlfile)
model_weights = os.path.join(MODELS_PATH,'efficientnetB0_checkpoint72_All') #TODO: update with the latest model

model = EfficientViT(config=config, channels=1280, selected_efficient_net=0)
model.load_state_dict(torch.load(model_weights))
model.eval()
model = model.cuda()
down_sample= AvgPool2d(kernel_size=2)

def generate_visualization(original_image):
    transformer_attribution = generate_relevance(model, original_image.unsqueeze(0).cuda()).detach()
    transformer_attribution = transformer_attribution.reshape(1, 1, 32, 32)
    transformer_attribution = down_sample(transformer_attribution)
    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=14, mode='bilinear')
    transformer_attribution = transformer_attribution.reshape(224, 224).cuda().data.cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (
            transformer_attribution.max() - transformer_attribution.min())
    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (
            image_transformer_attribution.max() - image_transformer_attribution.min())
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis


with open('test_summary_examples_test7.csv') as csv_file: #TODO: verify the right file name
    csv_reader = csv.DictReader(csv_file, delimiter=',')
    line_count = 0
    image_filenames_list =[]
    videos_preds =[]
    labels_list = []
    for row in csv_reader:
        if line_count==0:
            line_count += 1

        image_filenames_list.append(row["example_file_name"])
        videos_preds.append(row["video_prob"])
        labels_list.append(row["label"])
        line_count +=1

for i, file_name in enumerate(image_filenames_list):
    method = file_name.split('_')[0]
    full_path = os.path.join(EXAMPLES_DIR, file_name)
    image = Image.open(full_path)
    transform = create_base_transform(config['model']['image-size'])
    t_image = transform(image=cv2.imread(os.path.join(full_path)))['image']

    fig, axs = plt.subplots(1,2)
    axs[0].imshow(image);
    axs[0].axis('off');

    t_image = torch.tensor(np.asarray(t_image))
    t_image = np.transpose(t_image,(2, 0, 1)).float()

    pred_score = torch.sigmoid(model(t_image.unsqueeze(0).cuda()))

    label = 'Fake Image' if labels_list[i] == '1' else 'True Image'
    label = 'Label:          ' + label

    if pred_score.item() > 0.55 and labels_list[i] == '1':
        result = 'TP'
    elif pred_score.item() > 0.55 and labels_list[i] == '0':
        result = 'FP'
    elif pred_score.item() < 0.55 and labels_list[i] == '0':
        result = 'TN'
    else:
        result = 'FN'
    video_pred_score = 'Video Pred Score:   ' + str(videos_preds[i])[0:5]
    frame_pred_score = 'Frame Pred Score:  ' + str(pred_score.item())[0:5]
    Title = label + '\n' + 'Method:        ' + method + '\n' + video_pred_score + '\n' + frame_pred_score + '\nClassification:           ' + result

    image_vis = generate_visualization(t_image)

    axs[1].imshow(image_vis);
    axs[1].axis('off');

    plt.suptitle(Title,ha='left', size='medium', x=0.4, y=0.92)

    fig.savefig('samples/vis_' + image_filenames_list[i])
    plt.close(fig)



