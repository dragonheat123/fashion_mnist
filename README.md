# CNN for Fashion-MNIST Classification

### Data Augmentation
During training time with uniform distribution between range, using Keras ImageDataGenerator()

rotation +- 5 deg
translation +- 5% of 28x28 pixels
shear +- 0.05 deg
zoom +- 5% zoom
flip --> horizontal
```
train_transformations = ImageDataGenerator(
        rotation_range=5,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.05,
        horizontal_flip=True,
        fill_mode='nearest')
```
### CNN Model
```
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            [(None, 28, 28, 1)]  0                                            
__________________________________________________________________________________________________
batch_normalization (BatchNorma (None, 28, 28, 1)    4           input_1[0][0]                   

*Batch normalization to prevent the network from being fixated on a certain feature, 
by keeping the weights of the network from going too extreme
__________________________________________________________________________________________________
conv2d (Conv2D)                 (None, 26, 26, 128)  1280        batch_normalization[0][0]      

*large number of initial (3x3) filter to gather features for the model
__________________________________________________________________________________________________
dropout (Dropout)               (None, 26, 26, 128)  0           conv2d[0][0]           

*dropout between image layers to prevent overfitting to the training dataset
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 24, 24, 64)   73792       dropout[0][0]               

*second layer of filter to obtain higher level features
__________________________________________________________________________________________________
max_pooling2d (MaxPooling2D)    (None, 8, 8, 64)     0           conv2d_1[0][0]                   
__________________________________________________________________________________________________
average_pooling2d (AveragePooli (None, 8, 8, 64)     0           conv2d_1[0][0]                   
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 8, 8, 128)    0           max_pooling2d[0][0]              
                                                                 average_pooling2d[0][0]        
                                                                 
*combine both maxpooling + averagepooling in a layer to try to capture both sharp features 
(especially the edge of the images) and smooth out softer features (like logos in shirt)
__________________________________________________________________________________________________
flatten (Flatten)               (None, 8192)         0           concatenate[0][0]               

*convert output of filters into single feature vector
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 8192)         0           flatten[0][0]               

*again dropout to prevent overfitting
__________________________________________________________________________________________________
dense (Dense)                   (None, 128)          1048704     dropout_1[0][0]         

*dense layer to evaluate the feature vector and obtain accurate feature representation
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 10)           1290        dense[0][0]                 

*final softmax layer for output for catergorical classification
==================================================================================================
Total params: 1,125,070
Trainable params: 1,125,068
Non-trainable params: 2
```
### Accuracy
Test accuracy: 0.9217000007629395
