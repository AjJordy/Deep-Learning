from __future__ import print_function
import numpy as np # para computação numética menos intensiva
import os # para criar pastas
from matplotlib import pyplot as plt # para mostrar imagens
import tensorflow as tf # para redes neurais
from tensorflow.examples.tutorials.mnist import input_data # dataset

# criamos uma pasta para salvar o modelo
if not os.path.exists('tmp'): # se a pasta não existir
    os.makedirs('tmp') # cria a pasta

# Import MNIST data
data = input_data.read_data_sets("/tmp/data/", one_hot=False)

# baixa os dados na pasta criada e carrega os dados
# from tensorflow.examples.tutorials.mnist import input_data
# data = input_data.read_data_sets("tmp/", one_hot=False) # repare que não usamos vetores one-hot

# definindo constantes
lr = 0.01 # taxa de aprendizado
n_iter = 1000 # número de iterações de treino
batch_size = 128 # qtd de imagens no mini-lote (para GDE)
n_inputs = 28 * 28 # número de variáveis (pixeis)
n_l1 = 512 # número de neurônios da primeira camada
n_l2 = 512 # número de neurônios da segunda camada
n_outputs = 10 # número classes (dígitos)
logs_path = '/tmp/tensorflow_logs/example/' # caminho do log para o tensorboard


def fully_conected_layer(inputs, n_neurons, name_scope, activations=None):
    '''Adiciona os nós de uma camada ao grafo TensorFlow'''
    n_inputs = int(inputs.get_shape()[1]) # pega o formato dos inputs
    with tf.name_scope(name_scope):

        # define as variáveis da camada
        with tf.name_scope('Parameters'):
            W = tf.Variable(tf.truncated_normal([n_inputs, n_neurons]), name='Weights')
            b = tf.Variable(tf.zeros([n_neurons]), name='biases')

            tf.summary.histogram('Weights', W) # para registrar o valor dos W
            tf.summary.histogram('biases', b) # para registrar o valor dos b

        # operação linar da camada
        layer = tf.add(tf.matmul(inputs, W), b, name='Linear_transformation')

        # aplica não linearidade, se for o caso
        if activations == 'relu':
            layer = tf.nn.relu(layer, name='ReLU')

        # para registar a ativação na camada
        tf.summary.histogram('activations', layer)
        return layer


graph = tf.Graph()
with graph.as_default():

    # Camadas de Inputs
    with tf.name_scope('input_layer'):
        x_input = tf.placeholder(tf.float32, [None, n_inputs], name='images')
        y_input = tf.placeholder(tf.int64, [None], name='labels')

    # Camada 1
    l1 = fully_conected_layer(x_input, n_neurons=n_l1, name_scope='First_layer', activations='relu')

    # Camada 2
    l2 = fully_conected_layer(l1, n_neurons=n_l2, name_scope='Second_layer', activations='relu')

    # Camada de saída
    scores = fully_conected_layer(l2, n_neurons=n_outputs, name_scope='Output_layer') # logits

    # camada de erro
    with tf.name_scope('Error_layer'):
        error = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_input, logits=scores),
                               name='error')
        tf.summary.scalar('Cross_entropy', error) # para registrar a função custo

    # acuracia
    with tf.name_scope("Accuracy"):
        correct = tf.nn.in_top_k(scores, y_input, 1) # calcula obs corretas
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32)) # converta para float32
        tf.summary.scalar('Accuracy', accuracy) # para registrar a função custo

    # otimizador
    with tf.name_scope('Train_operation'):
        optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(error)

    # inicializador
    init = tf.global_variables_initializer()

    # para salvar o modelo treinado
    saver = tf.train.Saver()

    # para registrar na visualização
    summaries = tf.summary.merge_all() # funde todos os summaries em uma operação
    file_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph()) # para escrever arquivos summaries

# abrimos a sessão tf
with tf.Session(graph=graph) as sess:
    init.run() # iniciamos as variáveis

    # loop de treinamento
    for step in range(n_iter+1):

        avg_cost = 0.
        total_batch = int(data.train.num_examples/batch_size)
        # cria os mini-lotes
        x_batch, y_batch = data.train.next_batch(batch_size)

        # cria um feed_dict
        feed_dict = {x_input: x_batch, y_input: y_batch}

        # executa uma iteração de treino e calcula o erro
        l, summaries_str, _ = sess.run([error, summaries, optimizer], feed_dict=feed_dict)
        avg_cost += l / total_batch
        # a cada 10 iterações, salva os registros dos summaries
        if step % 10 == 0:
            file_writer.add_summary(summaries_str, step)
            print("Epoch:", '%04d' % (step+1), "cost=", "{:.9f}".format(avg_cost))


    print("Optimization Finished!")

    # Test model
    # Calculate accuracy
    print("Accuracy:", accuracy.eval({x_input: data.test.images, y_input: data.test.labels}))

    print("Run the command line:\n" \
          "--> tensorboard --logdir=/tmp/tensorflow_logs " \
          "\nThen open http://0.0.0.0:6006/ into your web browser")

file_writer.close() # fechamos o nó de escrever no disco.
