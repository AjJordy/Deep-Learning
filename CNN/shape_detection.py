import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.layers import GlobalAvgPool2D, Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Activation

num_imgs = 50000
img_size = 28
min_object_size = 4
max_object_size = 14
num_objects = 1

imgs = np.zeros((num_imgs,img_size,img_size), dtype=np.uint8) # Inicializa tela branca
bbox = np.zeros((num_imgs,num_objects,4),dtype=np.uint8)

# Criando o Dataset de imagens
for i_imgs in range(num_imgs): # for(int i=0; i<num_imgs; i++)
    for i_object in range(num_objects):
        w, h = np.random.randint(min_object_size,max_object_size,size = 2)
        y = np.random.randint(0,img_size-h)
        x = np.random.randint(0,img_size-w)
        imgs[i_imgs,x:x+w,y:y+h] = 255 # Quadrado preto
        bbox[i_imgs,i_object] = [x,y,w,h]

plt.imshow(imgs[0], origin='lower',cmap = 'Greys', extent=[0,img_size,0,img_size])
plt.show()
