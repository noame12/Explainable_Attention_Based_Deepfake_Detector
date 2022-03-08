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

**!Note:** Prior to running the _explain_model.py_ module, make sure to keep the same .csv file name or update the name in the _explain_model.py_ file (line 111).

- [ ] Run the explanation visualization process:
```
python explain_model.py
```

The output of the explanation process can be viewed in the _‘explanation’_ directory (created automatically)
![explanation process output](https://user-images.githubusercontent.com/93251301/157272590-774cf7d6-172d-48d0-8a44-1c3996f12507.png)


The results of the explanability process run on all examples in advance can be seen in the [visualization results drive](https://drive.google.com/drive/folders/1fxi-ilXykkq-RXwbNRtrwdicxKROrHae?usp=sharing) .

