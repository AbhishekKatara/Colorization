

```python
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import Activation, Dense, Dropout, Flatten, InputLayer
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.io import imsave
import numpy as np
import os
import random
import tensorflow as tf
```


```python
# Get images
# Change to '/data/images/Train/' to use all the 10k images
X = []
for filename in os.listdir('../Train/'):
    X.append(img_to_array(load_img('../Train/'+filename)))
X = np.array(X, dtype=float)

# Set up train and test data
split = int(0.95*len(X))
Xtrain = X[:split]
Xtrain = 1.0/255*Xtrain
```


```python
model = Sequential()
model.add(InputLayer(input_shape=(256, 256, 1)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
model.add(UpSampling2D((2, 2)))
model.compile(optimizer='rmsprop', loss='mse')
```


```python
# Image transformer
datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        horizontal_flip=True)

# Generate training data
batch_size = 10
def image_a_b_gen(batch_size):
    for batch in datagen.flow(Xtrain, batch_size=batch_size):
        lab_batch = rgb2lab(batch)
        X_batch = lab_batch[:,:,:,0]
        Y_batch = lab_batch[:,:,:,1:] / 128
        yield (X_batch.reshape(X_batch.shape+(1,)), Y_batch)

# Train model      
tensorboard = TensorBoard(log_dir="/output/beta_run")
model.fit_generator(image_a_b_gen(batch_size), callbacks=[tensorboard], epochs=50, steps_per_epoch=1)
```

    Epoch 1/50
    1/1 [==============================] - 30s - loss: 0.0668
    Epoch 2/50
    1/1 [==============================] - 27s - loss: 0.9944
    Epoch 3/50
    1/1 [==============================] - 27s - loss: 0.9935
    Epoch 4/50
    1/1 [==============================] - 27s - loss: 0.9972
    Epoch 5/50
    1/1 [==============================] - 19s - loss: 1.0213
    Epoch 6/50
    1/1 [==============================] - 27s - loss: 1.0179
    Epoch 7/50
    1/1 [==============================] - 27s - loss: 0.9896
    Epoch 8/50
    1/1 [==============================] - 27s - loss: 1.0032
    Epoch 9/50
    1/1 [==============================] - 27s - loss: 0.9969
    Epoch 10/50
    1/1 [==============================] - 19s - loss: 1.0180
    Epoch 11/50
    1/1 [==============================] - 27s - loss: 1.0048
    Epoch 12/50
    1/1 [==============================] - 27s - loss: 1.0371
    Epoch 13/50
    1/1 [==============================] - 27s - loss: 0.9935
    Epoch 14/50
    1/1 [==============================] - 27s - loss: 0.9862
    Epoch 15/50
    1/1 [==============================] - 19s - loss: 1.0013
    Epoch 16/50
    1/1 [==============================] - 27s - loss: 0.9972
    Epoch 17/50
    1/1 [==============================] - 27s - loss: 1.0105
    Epoch 18/50
    1/1 [==============================] - 27s - loss: 0.9957
    Epoch 19/50
    1/1 [==============================] - 27s - loss: 0.8023
    Epoch 20/50
    1/1 [==============================] - 19s - loss: 0.5173
    Epoch 21/50
    1/1 [==============================] - 27s - loss: 0.0163
    Epoch 22/50
    1/1 [==============================] - 27s - loss: 0.2679
    Epoch 23/50
    1/1 [==============================] - 27s - loss: 0.0067
    Epoch 24/50
    1/1 [==============================] - 27s - loss: 0.0127
    Epoch 25/50
    1/1 [==============================] - 19s - loss: 0.0114
    Epoch 26/50
    1/1 [==============================] - 27s - loss: 0.0116
    Epoch 27/50
    1/1 [==============================] - 27s - loss: 0.0204
    Epoch 28/50
    1/1 [==============================] - 27s - loss: 0.0060
    Epoch 29/50
    1/1 [==============================] - 27s - loss: 0.0076
    Epoch 30/50
    1/1 [==============================] - 19s - loss: 0.0170
    Epoch 31/50
    1/1 [==============================] - 27s - loss: 0.0121
    Epoch 32/50
    1/1 [==============================] - 27s - loss: 0.0152
    Epoch 33/50
    1/1 [==============================] - 27s - loss: 0.0111
    Epoch 34/50
    1/1 [==============================] - 27s - loss: 0.0215
    Epoch 35/50
    1/1 [==============================] - 19s - loss: 0.0120
    Epoch 36/50
    1/1 [==============================] - 27s - loss: 0.0070
    Epoch 37/50
    1/1 [==============================] - 27s - loss: 0.0117
    Epoch 38/50
    1/1 [==============================] - 27s - loss: 0.0144
    Epoch 39/50
    1/1 [==============================] - 27s - loss: 0.0136
    Epoch 40/50
    1/1 [==============================] - 19s - loss: 0.0131
    Epoch 41/50
    1/1 [==============================] - 27s - loss: 0.0112
    Epoch 42/50
    1/1 [==============================] - 27s - loss: 0.0165
    Epoch 43/50
    1/1 [==============================] - 27s - loss: 0.0060
    Epoch 44/50
    1/1 [==============================] - 27s - loss: 0.0146
    Epoch 45/50
    1/1 [==============================] - 19s - loss: 0.0066
    Epoch 46/50
    1/1 [==============================] - 27s - loss: 0.0149
    Epoch 47/50
    1/1 [==============================] - 27s - loss: 0.0069
    Epoch 48/50
    1/1 [==============================] - 27s - loss: 0.0066
    Epoch 49/50
    1/1 [==============================] - 27s - loss: 0.0161
    Epoch 50/50
    1/1 [==============================] - 19s - loss: 0.0094





    <keras.callbacks.History at 0x7f2964b31550>




```python
# Save model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
```


```python
# Test images
Xtest = rgb2lab(1.0/255*X[split:])[:,:,:,0]
Xtest = Xtest.reshape(Xtest.shape+(1,))
Ytest = rgb2lab(1.0/255*X[split:])[:,:,:,1:]
Ytest = Ytest / 128
print(model.evaluate(Xtest, Ytest, batch_size=batch_size))
```

    3/3 [==============================] - 2s
    0.901986598969



```python
# Change to '/data/images/Test/' to use all the 500 images
color_me = []
for filename in os.listdir('../Test/'):
	color_me.append(img_to_array(load_img('../Test/'+filename)))
color_me = np.array(color_me, dtype=float)
color_me = rgb2lab(1.0/255*color_me)[:,:,:,0]
color_me = color_me.reshape(color_me.shape+(1,))

# Test model
output = model.predict(color_me)
output = output * 128

# Output colorizations
for i in range(len(output)):
	cur = np.zeros((256, 256, 3))
	cur[:,:,0] = color_me[i][:,:,0]
	cur[:,:,1:] = output[i]
	imsave("result/img_"+str(i)+".png", lab2rgb(cur))
```

    /usr/local/lib/python3.6/site-packages/skimage/color/colorconv.py:985: UserWarning: Color data out of range: Z < 0 in 53398 pixels
      warn('Color data out of range: Z < 0 in %s pixels' % invalid[0].size)
    /usr/local/lib/python3.6/site-packages/skimage/io/_io.py:132: UserWarning: result/img_0.png is a low contrast image
      warn('%s is a low contrast image' % fname)
    /usr/local/lib/python3.6/site-packages/skimage/util/dtype.py:122: UserWarning: Possible precision loss when converting from float64 to uint8
      .format(dtypeobj_in, dtypeobj_out))
    /usr/local/lib/python3.6/site-packages/skimage/color/colorconv.py:985: UserWarning: Color data out of range: Z < 0 in 34189 pixels
      warn('Color data out of range: Z < 0 in %s pixels' % invalid[0].size)
    /usr/local/lib/python3.6/site-packages/skimage/io/_io.py:132: UserWarning: result/img_1.png is a low contrast image
      warn('%s is a low contrast image' % fname)
    /usr/local/lib/python3.6/site-packages/skimage/color/colorconv.py:985: UserWarning: Color data out of range: Z < 0 in 52095 pixels
      warn('Color data out of range: Z < 0 in %s pixels' % invalid[0].size)
    /usr/local/lib/python3.6/site-packages/skimage/color/colorconv.py:985: UserWarning: Color data out of range: Z < 0 in 64163 pixels
      warn('Color data out of range: Z < 0 in %s pixels' % invalid[0].size)
    /usr/local/lib/python3.6/site-packages/skimage/io/_io.py:132: UserWarning: result/img_3.png is a low contrast image
      warn('%s is a low contrast image' % fname)
    /usr/local/lib/python3.6/site-packages/skimage/color/colorconv.py:985: UserWarning: Color data out of range: Z < 0 in 24008 pixels
      warn('Color data out of range: Z < 0 in %s pixels' % invalid[0].size)
    /usr/local/lib/python3.6/site-packages/skimage/color/colorconv.py:985: UserWarning: Color data out of range: Z < 0 in 30082 pixels
      warn('Color data out of range: Z < 0 in %s pixels' % invalid[0].size)
    /usr/local/lib/python3.6/site-packages/skimage/io/_io.py:132: UserWarning: result/img_5.png is a low contrast image
      warn('%s is a low contrast image' % fname)
    /usr/local/lib/python3.6/site-packages/skimage/color/colorconv.py:985: UserWarning: Color data out of range: Z < 0 in 63477 pixels
      warn('Color data out of range: Z < 0 in %s pixels' % invalid[0].size)
    /usr/local/lib/python3.6/site-packages/skimage/color/colorconv.py:985: UserWarning: Color data out of range: Z < 0 in 34762 pixels
      warn('Color data out of range: Z < 0 in %s pixels' % invalid[0].size)

