from tkinter import *
from string import ascii_uppercase
from PIL import Image, ImageTk
from tkinter import filedialog
import numpy as np
from neuralNetwork import NeuralNetwork
from layer import Layer
from activation_layer import ActivationLayer
from activations import sigmoid, sigmoid_derivative, relu, relu_derivative
import socket
import pickle
from preprocess import preprocess
import pandas as pd

class App():
    client_socket = None

    def __init__(self, master=None):
        self.img = ""
        self.master = master
        self.panel = Label()
        self.letter = Label()
        self.output = None
        self.initialize_socket()
        self.initialize_gui()
    
    def initialize_socket(self):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((socket.gethostname(), 1249))

    def initialize_gui(self):
        self.master.title("Sign Language MNIST")
        self.master.geometry("700x500")
        self.master.resizable(width=True, height=True)
        self.displayHeader()
        self.displayButton()

    def send_recv(self, data):
        #Sending pixel values
        self.client_socket.send(pickle.dumps(data))
        print('pixel values sent successfully to the server!')

        #Receive the prediction
        out = self.client_socket.recv(100).decode('utf-8')

        #Converting hexadecimal to string
        out = [str(ord(x)) for x in out]
        self.output = out[0]
        print(f'received {self.output}')

    def displayHeader(self):
        #Title
        title = Label(self.master, text="Sign Language MNIST", font=("Helvetica", 30, "bold"))
        title.pack()

        #Subtitle
        subtitle = Label(self.master, text="Letter Prediction", font=("Helvetica", 22))
        subtitle.pack()

    def displayButton(self):
        l1 = Label(self.master, height=1)
        l1.pack()
        btn = Button(self.master, text='Open image', command=self.open_img)
        btn.pack()
        l2 = Label(self.master, height=1)
        l2.pack()
    
    def convert_to_pixels(self, img):
        img = img.convert('LA')
        img = img.resize((28, 28), Image.ANTIALIAS)
        pix = list(img.getdata())
        width, height = img.size
        pix = [pix[i * width:(i + 1) * width] for i in range(height)]
        p = [x[0] for sets in pix for x in sets]
        return p
    
    def open_img(self):
        #Open image
        x = self.openfilename()
        self.img = Image.open(x)
        
        #Convert image to pixels and preprocess...
        p = self.convert_to_pixels(self.img)
        p = pd.DataFrame(p).T
        p = preprocess(p)

        self.send_recv(p)

        #Display image on screen (250x250)
        self.img = Image.open(x)
        self.img = self.img.resize((250, 250), Image.ANTIALIAS)
        self.img = ImageTk.PhotoImage(self.img)

        self.panel['image'] = self.img
        self.panel['width'] = 250
        self.panel['height'] = 250
        self.panel['relief'] = "solid"
        self.panel["bd"] = 1
        self.panel.pack()

        l1 = Label(self.master, height=1)
        l1.pack()

        LETTERS = {letter: str(index) for index, letter in enumerate(ascii_uppercase, start=0)}
        keys = list(LETTERS.keys())
        values = list(LETTERS.values())

        pred = keys[values.index(str(self.output))]
        
        self.letter["text"] = "Predicted letter: " + pred
        self.letter["font"]=("Helvetica", 20)
        self.letter.pack()
        
    def openfilename(self):
        filename = filedialog.askopenfilename(title='"Open')
        return filename

#Main
root = Tk()
app = App(root)
root.mainloop()