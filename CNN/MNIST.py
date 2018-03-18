import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.layers import GlobalAveragePooling2D, Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Activation
from keras.datasets import cifar10, mnist
from keras.utils import np_utils


num_classes = 10
IMAGE_SIZE = 28

# importando dataset
(X_train,y_train),(X_test,y_test) = mnist.load_data()

print(X_train.shape) # dimensões do dataset

X_train = X_train.reshape(X_train.shape[0], IMAGE_SIZE, IMAGE_SIZE, 1)# 1 -> preto e branco
X_test = X_test.reshape(X_test.shape[0], IMAGE_SIZE, IMAGE_SIZE, 1)

# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
# X_train /= 255
# X_test /= 255
Y_train = np_utils.to_categorical(y_train, num_classes)
Y_test = np_utils.to_categorical(y_test, num_classes)

def modelo_top():
    model = Sequential()
    # (quantidade de filtros, (dimensao do filro), ativação, borda, dimensao da entrada)
    model.add(Conv2D(8,(3,3), activation='relu', padding='same', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)))
    model.add(Conv2D(8,(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    # model.add(Conv2D(64,(3,3), activation='relu', padding='same'))
    # model.add(Conv2D(64,(3,3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))

    model.summary()# Printa a rede
    return model

# modelo = modelo_top()

# Salvar modelo
# checkpoint = ModelCheckpoint('meuModeloTop.h5',
#                             monitor='val_loss',
#                             verbose=0,
#                             save_best_only = true,
#                             mode='auto')
#
# modelo.compile(loss='categorical_crossentropy',
#                optimizer=Adam(lr=0.0001),
#                metrics=['accuracy'])
#
# # (entrada, saída, quantidade de imagens por backpropagation, vezes que o processo será repetido)
# log = modelo.fit(X_train, Y_train,
#                  batch_size = 32,
#                  epochs = 10,
#                  validation_data = (X_test, Y_test),
#                  callbacks=[checkpoint],
#                  verbose=1)

# Carregar modelo salvo
modelo = load_model("meuModeloTop.h5")
scores = modelo.evaluate(X_test, Y_test, verbose=0)
print("Accuracy: % 2f%%" % (scores[1]*100))

# teste = cv2.imread("oito.png")
#

teste = X_test[0].reshape(1,28,28,1)

print(modelo.predict(teste))
pixels = teste
# reshape the array into 28 x 28 array (2 dimensional array)
pixels = pixels.reshape((28,28))

# Plot
plt.imshow(pixels, cmap='gray')
plt.show()
#plot_model
