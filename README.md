# Explainable Attention Based Deepfake Detector
The project consist a Video Deepfake detector based on hybrid EfficientNet CNN and Vision Transformer architecture. The model inference results can be analyzed and explained by rendering a (heatmap) visualization based on a Relevancy map calculated from the Attention layers of the Transformer, overlayed on the input face image.

![Explainable Attention Based Deepfake Detector](https://user-images.githubusercontent.com/93251301/157253542-47192d3e-c7f7-4aa0-bbd2-039738d8fba3.png)

In addition, the project enables to re-train and test the model performence and explainability with new parameters.

## How to Install  
- [ ] Clone the repository and move into it:
```
git clone https://github.com/noame12/Explainable_Attention_Based_Deepfake_Detector.git
cd Explainable_Attention_Based_Deepfake_Detector
```
- [ ] Setup Python environment using Conda:
```
conda env create --file environment.yml
conda activate explain_deepfakes
export PYTHONPATH=.
```
## Run the Explanation process on the deepfake detection model
**System requirements:**
To run the explanability process on more than 5 face images, a machine with Tesla T4 (or stronger) GPU is required. 

- [ ] Move to model explanation directory: 
```
cd  explain_model
```

- [ ] Create an input directory for the input face images:
```
mkdir examples
```
- [ ] Download the face images from the [samples](https://drive.google.com/drive/folders/1-JtWGMyd7YaTa56R6uYpjvwmUyW5q-zN?usp=sharing) drive into the newly created local _'examples'_ directory.

The samples drive contains 600 samples of face image extractions from 600 test videos. It consists of 100 images for each of the five deepfake methods – Face2Face, FaceShift, FaceSwap, NeuralTextures and Deepfakes, as well as 100 untouched real (aka Original) face images.

An exhaustive list of the face image files for running the explainability method is provided in [test_summary_All600_examples.csv](https://github.com/noame12/Explainable_Attention_Based_Deepfake_Detector/blob/master/Explain_model/test_summary_All600_examples.csv) file in the _'explain_model'_ directory. To run the test on a subset of the list, extract a customized list from the exhaustive list.

**!Note:** Make sure to keep the same .csv file name or update the name in the _explain_model.py_ file (line 111) prior to running the _explain_model.py_ module.

- [ ] Run the explanation visualization process:
```
python explain_model.py
```

The output of the explanation process can be viewed in the _‘explanation’_ directory (created automatically)
![explanation process output](https://user-images.githubusercontent.com/93251301/157272590-774cf7d6-172d-48d0-8a44-1c3996f12507.png)


The results of the explanability process run on all examples in advance can be seen in the [visualization results drive](https://drive.google.com/drive/folders/1fxi-ilXykkq-RXwbNRtrwdicxKROrHae?usp=sharing) .



## Test the deepfake detection model
The test module enables to test the performance of the deepfake detector. 
The input data to the model is the test (or verification) dataset of face images extracted from the fake and real video sequences. 
The test process generates four outputs:
-	Accuracy, AUC (Area Under Curve) and F1 scores of the classifier
-	ROC diagram
-	A .txt file with the classification results for each video sequence
-	A .csv list of face image files – one sample per each video.

**System requirements:**
To run the test process, a machine with **two** Tesla T4 (or stronger) GPUs is required. 


![Data flow](https://user-images.githubusercontent.com/93251301/157474640-5a6d5237-297d-42df-a7b3-0de615ff3a64.png)

### Get the data
- [ ] Download and extract the dataset:
[FaceForensic++](https://github.com/ondyari/FaceForensics/blob/master/dataset/)

The videos should be downloaded under _'/deep_fakes_exaplain/dataset'_ directory.

### Preprocess the data
To perform deepfake detection it is first necessary to identify and extract the faces from all the videos in the dataset.

- [ ] Detect the faces inside the videos:
```
cd preprocessing
```
```
python detect_faces.py --data_path /deep_fakes_exaplain/dataset --dataset: FACEFORENSICS
```
**!Note:** The default dataset for the detect_faces.py module is DFDC, therefore it is important to specify the --dataset parameter as described above.

The detected face boxes (coordinates) will be saved inside the "/deep_fakes_exaplain/dataset/boxes" folder.
![image](https://user-images.githubusercontent.com/93251301/157497703-050bf9c2-4962-49fe-b559-44f1ac3ab04e.png)


- [ ] Extract the detected faces obtaining the images:
```
python extract_crops.py --data_path deep_fakes_explain/dataset --output_path deep_fakes_explain/dataset/training_set
--dataset FACEFORENSIC
```
Repeat detection and extraction for all the different parts of your dataset. The --output_path parameter above is set to the training_set directory. You should repeat the process also for the validation_set and test_set directories.
The folders’ structure should look as follows: 
![image](https://user-images.githubusercontent.com/93251301/157499273-4c171cad-7163-4209-b7ac-e8d968cffa41.png)

Each (fake) method directory contain directories for all videos. Each video directory contain all face extraction files, for that video, in .png format.

```
    - training_set
        - Deepfakes
            - video_name_0
                0_0.png
                1_0.png
                ...
                N_0.png
            ...
            - video_name_K
                0_0.png
                1_0.png
                ...
                M_0.png
        - Face2Face
        - FaceShifter
        - FaceSwap
        - NeuralTextures
        - Original
    - validation_set
        ...
            ...
                ...
    - test_set
        ...
            ...
```
### Test the model
- [ ] Move into the test module folder:
```
cd model_test_train
```
- [ ] Run the following command for evaluating the deepfake detector model providing the pre-trained model path and the configuration file available in the config directory:
```
python test_model.py --model_path ../deep_fakes_explain/models/efficientnetB0_checkpoint72_All --config configs/explained_architecture.yaml
```
By default, the command will test on All datasets but you can customize the following parameters:
- --dataset: Which dataset to use (Deepfakes|Face2Face|FaceShifter|FaceSwap|NeuralTextures|Original|All)
- --workers: Number of data loader workers (default: 16)
- --frames_per_video: Number of equidistant frames for each video (default: 20)
- --batch_size: Prediction Batch Size (default: 12)

The results of the test process are saved in the _'results/tests'_ directory.

## Train the model
The train module enables to re-train the model with different parameters. Re-training may be desired for verifying or testing any thesis for improving model performance or explainability.

To evaluate a customized model trained from scratch with a different architecture, you need to edit the configs/explained_architecture.yaml file.

**System requirements:**
A machine with **two** Tesla T4 (or stronger) GPUs, CPU with 16 vCPUs and 100G RAM.

To train the model using my architecture configuration:
- [ ] Verify that you are in _‘model_test_train’_ directory
- [ ] Run the train module
```
python train_model.py --config configs/explained_architecture.yaml
```
By default the command will train on All method datasets but you can customize the following parameters:
- --num_epochs: Number of training epochs (default: 100)
- --workers: Number of data loader workers (default: 16)
- --resume: Path to latest checkpoint (default: none)
- --dataset: Which dataset to use (Deepfakes|Face2Face|FaceShifter|FaceSwap|NeuralTextures|All) (default: All)
- --max_videos: Maximum number of videos to use for training (default: all)
- --patience: How many epochs wait before stopping for validation loss not improving (default: 5)

## Credits
- The Deepfake Detector implementation is based on the [Hybrid EfficientNet Vision Transformer](https://github.com/davide-coccomini/Combining-EfficientNet-and-Vision-Transformers-for-Video-Deepfake-Detection) implementation.
- The explainability method is based on the  [Transformer MM Explainability](https://github.com/hila-chefer/Transformer-MM-Explainability) implementation.
