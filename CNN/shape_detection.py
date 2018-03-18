import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.layers import GlobalAvgPool2D, Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Activation
import math

# Parâmetros
num_imgs = 50000
img_size = 28
min_object_size = 4
max_object_size = 14
num_objects = 2

imgs = np.zeros((num_imgs,img_size,img_size), dtype=np.uint8) # Inicializa tela branca
bboxes = np.zeros((num_imgs,num_objects,4),dtype=np.uint8)

# Criando o Dataset de imagens
for i_imgs in range(num_imgs): # for(int i=0; i<num_imgs; i++)
    for i_object in range(num_objects):
        w, h = np.random.randint(min_object_size,max_object_size,size = 2)
        y = np.random.randint(0,img_size-h)
        x = np.random.randint(0,img_size-w)
        imgs[i_imgs,x:x+w,y:y+h] = 255 # Quadrado preto
        bboxes[i_imgs,i_object] = [x,y,w,h]

# Printar exemplos
'''
plt.imshow(imgs[0].T, origin='lower',cmap = 'Greys', extent=[0,img_size,0,img_size])
for bbox in bboxes[0]:
    plt.gca().add_patch(matplotlib.patches.Rectangle((bbox[0],bbox[1]),bbox[2],bbox[3],ec='r',fc='none'))
plt.show()
'''

# Normalizando Dataset
X = imgs.reshape(num_imgs,img_size,img_size,1) / 255
y = bboxes.reshape(num_imgs,-1) / img_size

# Separando Dataset
split = math.ceil(0.8 * num_imgs) # Separar 80% das imagens usadas para treino

train_X = X[:split]
test_X = X[split:]

train_y = y[:split]
test_y = y[split:]

# Backup para print
test_imgs = imgs[split:]
test_bboxes = bboxes[split:]

# Arquitetura da rede
def meu_modelo():
    modelo = Sequential()
    # qtd filtros, tamanho do filtro, ativação, borda, tamanho da entrada
    modelo.add(Conv2D(32,(3,3), activation='relu', padding='same',input_shape=(img_size,img_size,1)))
    modelo.add(MaxPooling2D(pool_size=(2,2)))
    modelo.add(Conv2D(64,(3,3), activation='relu'))
    # modelo.add(MaxPooling2D(pool_size=(2,2)))
    modelo.add(Conv2D(128,(3,3), activation='relu'))
    # modelo.add(MaxPooling2D(pool_size=(2,2)))
    modelo.add(Conv2D(128,(3,3), activation='relu'))
    modelo.add(Flatten())
    modelo.add(Dense(256,activation='relu'))
    modelo.add(Dropout(0.2))
    modelo.add(Dense(y.shape[-1]))
    return modelo

# Treinamento
model = meu_modelo()
model.compile(loss='mse', optimizer=Adam(lr=1.0e-4))
model.fit(train_X,train_y, epochs=10, validation_data=(test_X,test_y),verbose=2)

# Predição
pred_y = model.predict(test_X)
pred_bboxes = pred_y * img_size
pred_bboxes = pred_bboxes.reshape(len(test_imgs),num_objects,-1)

# Inserseção sobre união - Índice de Jaccard
def IOU(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1[0],bbox1[1],bbox1[2],bbox1[3]
    x2, y2, w2, h2 = bbox2[0],bbox2[1],bbox2[2],bbox2[3]
    w_I = min(x1 + w1, x2 + w2) - max(x1,x2) # Largura da interseção
    h_I = min(y1 + h1, y2 + h2) - max(y1,y2) # Altura da interseção
    if(w_I <= 0 or h_I <= 0):
        return 0
    I = w_I * h_I
    U = w1*h1 + w2*h2 - I
    return I/U

# Avaliar IOU
IOU_avg = 0
for i in range(len(test_imgs)):
    for pred_bbox, exp_bbox in zip(pred_bboxes[i],test_bboxes[i]): # Percorre uma lista de tuplas de preditos com experados
        IOU_avg += IOU(pred_bbox.T,exp_bbox.T)
IOU_avg /= len(test_imgs)

# Printar na tela
# TODO Copiar do material de aula
