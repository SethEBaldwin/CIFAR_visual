
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

model = keras.Sequential([
    keras.layers.Conv2D(filters=32,kernel_size=3,strides=1,padding='same',
		input_shape=(32, 32, 3),activation='elu',
		kernel_regularizer=keras.regularizers.l2(1e-4)),
	keras.layers.BatchNormalization(),
	keras.layers.Conv2D(filters=32,kernel_size=3,strides=1,padding='same',
		activation='elu',kernel_regularizer=keras.regularizers.l2(1e-4)),
	keras.layers.BatchNormalization(),
	keras.layers.MaxPool2D(),
	keras.layers.Dropout(.2),
	keras.layers.Conv2D(filters=64,kernel_size=3,strides=1,padding='same',
		activation='elu',kernel_regularizer=keras.regularizers.l2(1e-4)),
	keras.layers.BatchNormalization(),
	keras.layers.Conv2D(filters=64,kernel_size=3,strides=1,padding='same',
		activation='elu',kernel_regularizer=keras.regularizers.l2(1e-4)),
	keras.layers.BatchNormalization(),
	keras.layers.MaxPool2D(),
	keras.layers.Dropout(.3),
	keras.layers.Conv2D(filters=128,kernel_size=3,strides=1,padding='same',
		activation='elu',kernel_regularizer=keras.regularizers.l2(1e-4)),
	keras.layers.BatchNormalization(),
	keras.layers.Conv2D(filters=128,kernel_size=3,strides=1,padding='same',
		activation='elu',kernel_regularizer=keras.regularizers.l2(1e-4)),
	keras.layers.BatchNormalization(),
	keras.layers.MaxPool2D(),
	keras.layers.Dropout(.4),
	keras.layers.Flatten(),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.summary()

adam = tf.keras.optimizers.Adam(lr=.001,decay=1e-5)
model.compile(optimizer=adam,loss='sparse_categorical_crossentropy', metrics=['accuracy'])

datagen = keras.preprocessing.image.ImageDataGenerator(
	rotation_range=15,width_shift_range=0.1,height_shift_range=0.1,horizontal_flip=True
	)
datagen.fit(x_train)

history = model.fit(
	datagen.flow(x_train, y_train, batch_size=32), 
	batch_size=32, epochs=128, validation_data = (x_test,y_test)
	)

test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)

model.save('cifar_model')
print('Model saved')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
