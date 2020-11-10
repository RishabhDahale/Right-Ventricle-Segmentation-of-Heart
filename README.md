# Right-Ventricle-Segmentation-of-Heart

This repo is the work done in assignment for the course EE 782 Advanced Machine Learning at IIT Bombay. We have studies the effect of changing various hyperparameters and other related things during training to check it's effect on training process and level of generalization the model is able to achieve. The following sections give a glimpse of work done. More details of the results can be found in the report. Results obtained during our runs can be found at this [drive link](https://drive.google.com/drive/folders/1nsCYAdpSwXSHBMecZQRnpJm5Rd9QFJCq?usp=sharing)

# Dataset and Preprocessing Steps
Dataset used in this analysis was from Right Ventricle Segmentation Challenge (RVSC) which was hosted in March 2012. The dataset was collected from June 2008 to August 2008. All patients were above the age of 18 and were free from any cardiac conditions. The total of 48 patients participated in this study. The average age ($\pm$ standard deviation) of the participants was 52:1($\pm$ 18:1) years. The cardiac MR examinations were performed with repeated breath-holds for a period of 10-15 s. All MR images were taken with a total of 10-14 contiguous cine short axis slices from the base to the top of the ventricles. The images are of sizes 256x216 pixels, and further 20 images were taken per cardiac cycle. More details about the dataset can be found on [their website](http://rvsc.projets.litislab.fr/)<br>
To help model generalize the results well, we have performed following transforms on the raw input image
* Contrast Limited Adaptive Histogram Equilization (CLAHE) was applied to improve the contrast of the images which helped in better distinguising of the boundaries and various other things
* Random Rotate: Images were randomly rotated about their center with a probability of 0.5. The rotation angle was sampled from a uniform distribution in -360 to 360 degree
* Random Crop: Images were randomly cropped in 192x192 size images. This acted like a linear translation and helped the model for better generalization <br>
Script for dataloading and these proprecessing steps can be found in [this script in utility folder](https://github.com/RishabhDahale/Right-Ventricle-Segmentation-of-Heart/blob/main/utility/dataset.py).

# Baseline Model
We had selected the baseline model as U-net with depth of 4. As all the experiments were run on google colab we had to keep the depth small to trade it with training speed and batch size. Model was trained for 300 epochs with adam optimizer. Initial learning rate was kept at $2 \times 10^{-3}$. The train-validation split of the data was kept as 80-20. Step lr scheduler was used which reduced the learning rate to 10% of it's value after every 100 epochs.<br>
For baseline we used the loss function as simple average of dice loss and inverse dice loss. Inverse dice is the dice loss on the backgroung instead of foreground. 

# Variations experimented
We tried to change the following parameters from the baseline model
* Optimizer - We tried adam, SGD and RMSProp optimizers. SGD was used with a momentum of 0.95
* LR Scheduler - We tried 2 different schedulers. Step LR and Exponential LR. For Exponential LR we tried 2 different decay factors: 0.97 and 0.99
* Effect of changing the train-validation-split was also studied. Effect of varying the training data on different optimizers and lr schedulers is reported
* Weight Decay - Effect on training process by varying weight decay
* Effect of varying the dropout probability on model generalization
* Loss function - Instead of simple average of dice loss and inverse dice loss, we took weighted average of dice loss and inverted dice loss. Results on different weights are reported
* Mode of upsampling - In the upsampling layer of the unet, upsampling can be done with 2 methods: conv transpose 2d and bilinear interpolation + 1x1 convolution. Effect of both the method on model learning is studied and reported
* Effect of using L1 and L1L2 regularizers is also studies<br>

# Code breakage
The main script which needs to be called is train.py. It accepts various command line parameters, whose default values can be found in [utility/parameters.py](https://github.com/RishabhDahale/Right-Ventricle-Segmentation-of-Heart/blob/main/utility/parameters.py).<br>
Another method to give the input parameters is through a config file. Config file for the baseline model is present in **configurations** folder. To run the script with config file as input use the command
```
python train.py <path_to_config_file>
```
