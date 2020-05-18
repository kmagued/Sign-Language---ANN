import socket
import pickle
from neuralNetwork import NeuralNetwork
from layer import Layer
from activation_layer import ActivationLayer
from activations import relu, relu_derivative, sigmoid, sigmoid_derivative
import numpy as np

class Server:
    def __init__(self):
        self.server_socket = None
        self.createServer()

    def createServer(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((socket.gethostname(), 1249))
        self.server_socket.listen(5)

    def predict(self, data):
        nn = NeuralNetwork()
        l1 = Layer(56, 54)
        l2 = Layer(54, 25)

        nn.add(l1)
        nn.add(ActivationLayer(relu, relu_derivative))
        nn.add(l2)
        nn.add(ActivationLayer(sigmoid, sigmoid_derivative))

        l1.weights = np.load('weights1.npy')
        l2.weights = np.load('weights2.npy')

        l1.bias = np.load('bias1.npy')
        l2.bias = np.load('bias2.npy')

        out = nn.predict(data)
        pred = np.argmax(out)

        return pred

    def receive_data(self):
        while True:
            print('Waiting for connection...')
            c, addr = self.server_socket.accept()
            print(f'Got connection from {addr}')

            while True:
                data = c.recv(4096)
                print('Received pixel values!')
                d = pickle.loads(data)
                    
                if data:
                    pred = self.predict(d)
                    print(f'Sending prediction: {pred}')
                    c.send(pred)
                
                else:
                    break

#Main
s = Server()
s.receive_data()