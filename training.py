import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,BatchNormalization
from keras.layers import Conv2D,MaxPooling2D
import os

num_classes=7
img_rows,img_cols=48,48
batch_size=32
##r here changes the backward to forward slash
train_data=r'C:\Users\varsh\OneDrive\Desktop\facial-expression\train'
validation_data=r'C:\Users\varsh\OneDrive\Desktop\facial-expression\validation'

##here rescale is used for scaling down the data
##We zoom the data by 30%
##Shear range is kept 0.3
##width shift and height shift the photo by 40%
train_data_gen=ImageDataGenerator(rescale=1./255,
                                  rotation_range=30,shear_range=0.3,
                                  zoom_range=0.3,width_shift_range=0.4,height_shift_range=0.4,
                                  horizontal_flip=True,vertical_flip=True,fill_mode='nearest')
validation_data_gen=ImageDataGenerator(rescale=1./255)
##these generator are used to train the data
train_generator= train_data_gen.flow_from_directory(train_data,color_mode='grayscale',target_size=(img_rows,img_cols),
                                                    batch_size=batch_size,class_mode='categorical',shuffle=True)
validation_generator=validation_data_gen.flow_from_directory(validation_data,color_mode='grayscale',target_size=(img_rows,img_cols),
                                                 batch_size=batch_size,class_mode='categorical',shuffle=True)
##We use softmax as we have more than 2 outputs. But sigmoid is used for binary outputs.
##Training CNN model
##Block1 
model=Sequential()
model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(img_rows,img_cols,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(img_rows,img_cols,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))##it is used for preventing overfitting of the data.

##model2

model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
##model3

model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
##model4

model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0,2))

##model 5
model.add(Flatten())
model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

##model 6
model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

##model 7
model.add(Dense(num_classes,kernel_initializer='he_normal'))
model.add(Activation('softmax')) ## output layer

##print(model.summary())

from keras.optimizers import RMSprop,SGD,Adam
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau

checkpoint= ModelCheckpoint(r'C:\Users\varsh\OneDrive\Desktop\facial-expression\Emotion_little_vgg.h5',
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True,
                             verbose=1)

earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=3,
                          verbose=1,
                          restore_best_weights=True
                          )

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=3,
                              verbose=1,
                              min_delta=0.0001)

callbacks = [earlystop,checkpoint,reduce_lr]

model.compile(loss='categorical_crossentropy',
              optimizer = Adam(lr=0.001),
              metrics=['accuracy'])

nb_train_samples = 24176
nb_validation_samples = 3006
epochs=25

history=model.fit_generator(
                train_generator,
                steps_per_epoch=nb_train_samples//batch_size,
                epochs=epochs,
                callbacks=callbacks,
                validation_data=validation_generator,
                validation_steps=nb_validation_samples//batch_size)


