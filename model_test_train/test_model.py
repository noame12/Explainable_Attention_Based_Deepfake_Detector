import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
import os
from shutil import copyfile
import cv2
import numpy as np
import torch

from sklearn.metrics import f1_score
from albumentations import Compose, RandomBrightnessContrast, \
    HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, \
    ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur, Rotate

from transforms.albu import IsotropicResize

from utils import get_method, check_correct, resize, shuffle_dataset, get_n_params
import torch.nn as nn
from functools import partial
from evit_model10 import EfficientViT
from multiprocessing.pool import Pool
from progress.bar import Bar
import pandas as pd
from tqdm import tqdm
from multiprocessing import Manager
from utils import custom_round, custom_video_round
import yaml
import argparse
import csv


RESULTS_DIR = "results"
BASE_DIR = "../deep_fakes_explain"
DATA_DIR = os.path.join(BASE_DIR, "dataset")
TEST_DIR = os.path.join(DATA_DIR, "validation_set")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "tests")

TEST_LABELS_PATH = os.path.join(BASE_DIR, "dataset/dfdc_test_labels.csv")



if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def create_base_transform(size):
    return Compose([
        IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
    ])

def save_roc_curves(correct_labels, preds, model_name, accuracy, loss, f1):
  plt.figure(1)
  plt.plot([0, 1], [0, 1], 'k--')

  fpr, tpr, th = metrics.roc_curve(correct_labels, preds)

  model_auc = auc(fpr, tpr)


  plt.plot(fpr, tpr, label="Model_"+ model_name + ' (area = {:.3f})'.format(model_auc))

  plt.xlabel('False positive rate')
  plt.ylabel('True positive rate')
  plt.title('ROC curve')
  plt.legend(loc='best')
  plt.savefig(os.path.join(OUTPUT_DIR, model_name +  "_" + opt.dataset + "_acc" + str(accuracy*100) + "_loss"+str(loss)+"_f1"+str(f1)+".jpg"))
  plt.clf()

def read_frames(video_path, videos):
    
    # Get the video label based on dataset selected
    method = get_method(video_path, DATA_DIR)
    if "Original" in video_path:
        label = 0.
    elif method == "DFDC":
        test_df = pd.DataFrame(pd.read_csv(TEST_LABELS_PATH))
        video_folder_name = os.path.basename(video_path)
        video_key = video_folder_name + ".mp4"
        label = test_df.loc[test_df['filename'] == video_key]['label'].values[0]
    else:
        label = 1.

    selected_frames =[]

    # Calculate the interval to extract the frames
    frames_number = len(os.listdir(video_path))
    frames_interval = int(frames_number / opt.frames_per_video)
    frames_paths = os.listdir(video_path)
    frames_paths_dict = {}

    # Group the faces with the same index, reduce probabiity to skip some faces in the same video
    for path in frames_paths:
        for i in range(0,3):
            if "_" + str(i) in path:
                if i not in frames_paths_dict.keys():
                    frames_paths_dict[i] = [path]
                else:
                    frames_paths_dict[i].append(path)

    # Select only the frames at a certain interval
    if frames_interval > 0:
        for key in frames_paths_dict.keys():
            if len(frames_paths_dict) > frames_interval:
                frames_paths_dict[key] = frames_paths_dict[key][::frames_interval]
            
            frames_paths_dict[key] = frames_paths_dict[key][:opt.frames_per_video]

        for key in frames_paths_dict:
            selected_frames.append(frames_paths_dict[key])


    # Select N images from the collected frames
    video = {}
    for key in frames_paths_dict.keys():
        for index, frame_image in enumerate(frames_paths_dict[key]):
            transform = create_base_transform(config['model']['image-size'])
            image = transform(image=cv2.imread(os.path.join(video_path, frame_image)))['image']
            if len(image) > 0:
                if key in video:
                    video[key].append(image)
                else:
                    video[key] = [image]
    videos.append((video, label, video_path,selected_frames))





# Main body
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--workers', default=16, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--model_path', default='', type=str, metavar='PATH',
                        help='Path to model checkpoint (default: none).')
    parser.add_argument('--dataset', type=str, default='All',
                        help="Which dataset to use (Deepfakes|Face2Face|FaceShifter|FaceSwap|NeuralTexture|All)")
    parser.add_argument('--max_videos', type=int, default=-1, 
                        help="Maximum number of videos to use for training (default: all).")
    parser.add_argument('--config', type=str, 
                        help="Which configuration to use. See into 'config' folder.")
    parser.add_argument('--efficient_net', type=int, default=0, 
                        help="Which EfficientNet version to use (0 or 7, default: 0)")
    parser.add_argument('--frames_per_video', type=int, default=20,
                        help="How many equidistant frames for each video (default: 20)")
    parser.add_argument('--batch_size', type=int, default=12,
                        help="Batch size (default: 12)") #Todo: I canged the default value to 1
    
    opt = parser.parse_args()
    print(opt)

    with open(opt.config, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)

    if opt.efficient_net == 0:
        channels = 1280
    else:
        channels = 2560

    if os.path.exists(opt.model_path):
        model = EfficientViT(config=config, channels=channels, selected_efficient_net = opt.efficient_net)
        model.load_state_dict(torch.load(opt.model_path))
        parallel_net = nn.DataParallel(model, device_ids=[0,1]) #Fixme: parallel net added in
        parallel_net = parallel_net.to(0)
        parallel_net.eval()
        # model.eval()
        # model = model.cuda()
    else:
        print("No model found.")
        exit()

    model_name = os.path.basename(opt.model_path)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    
    preds = []
    mgr = Manager()
    paths = [] # a list of paths to videos
    videos = mgr.list()


    if opt.dataset == 'All':
        folders = ["Original", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures", "Deepfakes"]

    else:
        folders = [opt.dataset]

    for folder in folders:
        method_folder = os.path.join(TEST_DIR, folder)  
        for index, video_folder in enumerate(os.listdir(method_folder)):
            paths.append(os.path.join(method_folder, video_folder)) # populate the list of paths to videos
      
    with Pool(processes=opt.workers) as p:
        with tqdm(total=len(paths)) as pbar:
            for v in p.imap_unordered(partial(read_frames, videos=videos),paths):
                pbar.update()

    video_names = np.asarray([row[2] for row in videos]) # full path to videos
    correct_test_labels = np.asarray([row[1] for row in videos])
    selected_frames = np.asarray([row[3] for row in videos])
    videos = np.asarray([row[0] for row in videos]) # dictionary of images per video

    preds = []


    print ('video_names', video_names)

    bar = Bar('Predicting \n', max=len(videos))

    f = open(opt.dataset + "_" + model_name + "_labels.txt", "w+")

    max_min_probs = []
    max_min_images = []
    for index, video in enumerate(videos):
        video_faces_preds = []
        video_name = video_names[index]
        f.write(video_name)
        for key in video:
            faces_preds = []
            video_faces = video[key]
            for i in range(0, len(video_faces), opt.batch_size):
                faces = video_faces[i:i+opt.batch_size]
                faces = torch.tensor(np.asarray(faces))
                if faces.shape[0] == 0:
                    continue
                faces = np.transpose(faces, (0, 3, 1, 2))

                faces = faces.float().to(0)
                pred = parallel_net(faces)
                
                scaled_pred = []
                for idx, p in enumerate(pred):
                    scaled_pred.append(torch.sigmoid(p))
                faces_preds.extend(scaled_pred)

            current_faces_pred = sum(faces_preds)/len(faces_preds)
            face_pred = current_faces_pred.cpu().detach().numpy()[0]
            f.write(" " + str(face_pred))
            video_faces_preds.append(face_pred)

        # Following code samples face images for visualization - one image for each video.
        # If video classified fake than sampling the frame with the highest prediction score.
        # Otherwise, if video classified real,  sampling the frame with the lowest prediction score.

        if max(video_faces_preds) > 0.55:
            video_faces_max_min = max(faces_preds)
            key = video_faces_preds.index(max(video_faces_preds))
        else:
            video_faces_max_min = min(faces_preds)
            key = video_faces_preds.index(min(video_faces_preds))

        max_min_index = faces_preds.index(video_faces_max_min)
        video_faces_max_min = video_faces_max_min.detach().cpu()
        # key = video_faces_preds.index(max(video_faces_preds))
        max_min_image = selected_frames[index][key][max_min_index]
        max_min_probs.append(np.asarray(video_faces_max_min))
        max_min_images.append(max_min_image)

        bar.next()
        if len(video_faces_preds) > 1:
            video_pred = custom_video_round(video_faces_preds)

        else:
            video_pred = video_faces_preds[0]


        preds.append([video_pred])

        
        f.write(" --> " + str(video_pred) + "(CORRECT: " + str(correct_test_labels[index]) + ")" +"\n")
        
    f.close()
    dst_name = os.path.join(OUTPUT_DIR,f.name)
    copyfile(f.name, dst_name)
    bar.finish()


    # csv header
    header = ['video_name', 'label', 'video_prob', 'high_low_prob', 'high_low_frame_path']
    summary = np.asarray(video_names.reshape(len(video_names),-1))
    summary = np.append(summary,correct_test_labels.reshape(len(correct_test_labels),-1), axis=1)
    summary = np.append(summary,preds, axis=1)
    summary = np.append(summary, np.asarray(max_min_probs), axis=1)
    summary = np.append(summary,np.asarray(max_min_images).reshape(len(max_min_images),-1), axis=1)


    with open('test_summary'+ '_' + opt.dataset + '.csv', 'w', encoding='UTF8', newline='') as out:
        writer = csv.writer(out)
        writer.writerow(header)  # write the header
        writer.writerows(summary) # write multiple rows

    dst_csv = os.path.join(OUTPUT_DIR, out.name)
    copyfile(out.name, dst_csv)


    loss_fn = torch.nn.BCEWithLogitsLoss()
    tensor_labels = torch.tensor([[float(label)] for label in correct_test_labels])
    tensor_preds = torch.tensor(preds)


    loss = loss_fn(tensor_preds, tensor_labels).numpy()

    #accuracy = accuracy_score(np.asarray(preds).round(), correct_test_labels)
    accuracy = accuracy_score(custom_round(np.asarray(preds)), correct_test_labels)

    f1 = f1_score(correct_test_labels, custom_round(np.asarray(preds)))
    print(model_name, "Test Accuracy:", accuracy, "Loss:", loss,  "F1", f1)
    save_roc_curves(correct_test_labels, preds, model_name, accuracy, loss, f1)
    # print("Model AUC", model_auc)
