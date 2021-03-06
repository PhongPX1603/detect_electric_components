# Detect Electric Components


## Installation
### Installing from source

For normal training and evaluation we should install the package from source using colab environment.

```bash
git clone https://github.com/PhongPX1603/detect_electric_components.git
cd detect_electric_components/
pip install pytorch-ignite
pip install git+https://github.com/aleju/imgaug.git
```

#### Download pretrained weights

```bash
https://drive.google.com/file/d/1e4ageUTMBVpGc3Nrdrr1cEzMS6Gnsj_M/view?usp=sharing
```

## Project Structure
```
    detect_electric_components
                    |
                    ├── config
                    |	  ├── electric_labelme_training.yaml      #train with label json file
                    |	  ├── electric_labelme_testing.yaml
                    |     └── txt_training.yaml                   #train with label txt file
                    |
                    ├── flame
                    |	  ├── core 
                    |     |     ├── data 
                    |     |     |     ├── electric_components.py
                    |     |     |     └── visualize.py
                    |     |     |
                    |     |     ├── engine
                    |     |     |     ├── evaluator.py
                    |     |     |     └── trainer.py
                    |     |     |
                    |     |     ├── loss
                    |     |     |     ├── compute_iou.py
                    |     |     |     ├── generate_anchors.py
                    |     |     |     ├── loss.py
                    |     |     |     └── yolov3_loss.py
                    |     |     |
                    |     |     └── model
                    |     |           ├── darknet53.py
                    |     |           └── model.py
                    |     |     
                    |     |     
                    |	  └── handle               
                    |           ├── metrics
                    |           |     ├── loss
                    |           |     |     ├── loss.py
                    |           |     |     └── yolov3_loss.py
                    |           |     |
                    |           |     ├── mean_average_precision
                    |           |     |     ├── evaluator.py
                    |           |     |     └── mean_average_precision.py
                    |           |     |
                    |           |     └── metrics.py
                    |           |
                    |           ├── checkpoint.py
                    |           ├── early_stopping.py
                    |           ├── lr_scheduler.py
                    |           ├── metric_evaluator.py
                    |           ├── region_predictor.py
                    |           ├── screenlogger.py
                    |           └── terminate_on_nan.py
                    |
                    ├── inference
                    |	  ├── config.yaml
                    |	  ├── darknet53.py
                    |	  ├── predictor.py
                    |	  ├── real_time_inference.py
                    |     └── utils.py
                    |
                    ├── __main__.py
                    ├── module.py
                    └── utils.py
```
## Dataset
* Data more 200 "Electric Components" images.
* Data divided into 2 parts: train and valid folders.

| Name  | Train | Valid | Test | Label's Format |
| ---   | ---         |     ---      |  --- |   --- |
| Electric Components | 167 |  39    |  ---   | json or txt    |

#### Use augumentation technical to make variety the dataset. Use library: "imgaug.augmenters". Some augumentations I use:
Install imgaug package ```pip install git+https://github.com/aleju/imgaug.git```
- iaa.MotionBlur()      # make image to Blur a little
- iaa.ChangeColorTemperature()      # change color of image follow random temperature 
- iaa.GaussianBlur(sigma=(0, 1))        # Augmenter to blur images using gaussian kernels. Sigma ís blur level
- iaa.Grayscale(alpha=(0.0, 1.0))       # Augmenter to convert images to their grayscale versions. The alpha value of the grayscale image when overlayed over the                                           old image. A value close to 1.0 means, that mostly the new grayscale image is visible and opposite with 0.0 means.
- iaa.Add(value=(-50, 50), per_channel=True)        # Add a value to all pixels in an image and that value between -50 to 50. Per_channel=True mean add for all                                                         pixel on image. 
- iaa.Fliplr(p=0.5)     # Flip/mirror input images horizontally. p=0.5 is probality apply.
- iaa.Flipud(p=0.5)     # Flip/mirror input images vertically. p=0.5 is probality apply.
- iaa.Crop(percent=(0, 0.1))        # Crop images, i.e. remove columns/rows of pixels at the sides of images. The number of pixels to crop on each side of the                                           image given as a fraction of the image height/width. 0.1 equivalent 10% of image will be croped on 4 edges.
- iaa.Pad(percent=(0, 0.1), keep_size=False)        # Pad images, i.e. adds columns/rows of pixels to them. The number of pixels to pad on each side of the image                                                       given as a fraction of the image height/width. 0.1 equivalent 10% of image will be paded on 4 edges.

## Model & Metrics
- I used YOLOv3 and also use pretrain model trained with "Electric Components" dataset to greatly reduce training time.
- In training process, I used "Learning Rate Schedule" and "Early Stopping" to adjust learning rate follow loss value (3 epochs) and stop training when loss unimprove passing some epochs (10 epochs).

## How to Run
### Clone github
* Run the script below to clone my github.
```
git clone https://github.com/PhongPX1603/detect_electric_components.git
```

### Training
* Trained by pytorch-ignite framework. Install: ```pip install pytorch-ignite```
* Dataset structure
```
dataset
    ├── train
    │   ├── img1.jpg
    │   ├── img1.json
    |   ├── img2.jpg
    |   ├── img2.json
    │   └── ...
    │   
    └── valid
        ├── img1.jpg
        ├── img1.json
        ├── img2.jpg
        ├── img2.json
        └── ...
```
* Change your direct of dataset folder in ```config/electric_labelme_training.yaml```
* Run the script below to train the model. Specify particular name to identify your experiment:
```python -m flame configs/electric_labelme_training.yaml```

### Test
* Change your direct of test dataset folder and weights file in ```config/electric_labelme_testing.yaml```
* Run the script below to test the model:
```python -m flame configs/electric_labelme_testing.yaml```


## Inference
### Installing to inference
```bash
pip install -r requirements.txt
```
* You can use this script to make inferences on particular folder
* Result are saved at <output/img.jpg> if type inference is 'image' or <video-output.mp4> with 'video or webcam' type.
```
cd inference
python real_time_inference.py --type-inference 'image' --input-dir <image dir> --video-output <video_output.mp4>
                                               'video'             <video dir>
                                               'webcam'            0
```
![yolov3](https://user-images.githubusercontent.com/86842861/143871002-b2516c01-b3d2-4d2a-b1fe-62bfb28bcb47.gif)

## Feature
* YOLOv4: https://github.com/PhongPX1603/components_yolov4

## Acknowledgements
* https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/YOLOv3/model.py

## Contributor
*Xuan-Phong Pham*
