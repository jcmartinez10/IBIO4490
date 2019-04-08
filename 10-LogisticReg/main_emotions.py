import numpy as np
import urllib.request
import zipfile
import os
import getopt
import sys
import pickle
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.utils.fixes import signature




if not os.path.isfile('./fer2013.csv'):
    if not os.path.isfile('./fer2013.zip'):
        urllib.request.urlretrieve('http://bcv001.uniandes.edu.co/fer2013.zip', './fer2013.zip')
    zip_ref = zipfile.ZipFile('./fer2013.zip', 'r')
    zip_ref.extractall('./')
    zip_ref.close()
    
opts, args = getopt.getopt(
    sys.argv[1:],
    't:d',
    ['test', 'demo'],
)


def softmax(x):
    return np.exp(x - x.max()) / np.sum(np.exp(x - x.max()))

def get_data():
    # angry, disgust, fear, happy, sad, surprise, neutral
    with open("fer2013.csv") as f:
        content = f.readlines()

    lines = np.array(content)
    num_of_instances = lines.size
    print("number of instances: ",num_of_instances)
    print("instance length: ",len(lines[1].split(",")[1].split(" ")))

    x_train, y_train, x_test, y_test = [], [], [], []

    for i in range(1,num_of_instances):
        emotion, img, usage = lines[i].split(",")
        pixels = np.array(img.split(" "), 'float32')
        emotion = int(emotion)
        if 'Training' in usage:
            y_train.append(emotion)
            x_train.append(pixels)
        elif 'PublicTest' in usage:
            y_test.append(emotion)
            x_test.append(pixels)

    #------------------------------
    #data transformation for train and test sets
    x_train = np.array(x_train, 'float64')
    y_train = np.array(y_train, 'float64')
    x_test = np.array(x_test, 'float64')
    y_test = np.array(y_test, 'float64')

    x_train /= 255 #normalize inputs between [0, 1]
    x_test /= 255

    x_train = x_train.reshape(x_train.shape[0], 48, 48)
    x_test = x_test.reshape(x_test.shape[0], 48, 48)
    y_train = y_train.reshape(y_train.shape[0], 1)
    y_test = y_test.reshape(y_test.shape[0], 1)

    global x_score
    global y_score

    x_score=x_test
    y_score=y_test

    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    return x_train, y_train, x_test, y_test

class Model():
    def __init__(self):
        params = 48*48 # image reshape
        out = 7 # all labels
        
        lr_str = input("Learning rate: ")
        if len(lr_str) == 0 :
            lr=0.0001# Change if you want
            print('Set to default ('+str(lr)+')')
        else:
            lr=float(lr_str)
        self.lr = lr
        self.W = np.random.randn(params, out)
        self.b = np.random.randn(out)

    def forward(self, image):
        image = image.reshape(image.shape[0], -1)
        out = np.dot(image, self.W) + self.b
        return out

    def compute_loss(self, pred, gt): #son arrays
        
        probs = softmax(pred)
        gt=gt.astype(int)
        correct_logprobs = -np.log(probs[range(pred.shape[0]),gt])
        reg_loss = 0.5*0.001*np.sum(self.W*self.W)
        J=np.sum(correct_logprobs)/pred.shape[0]+reg_loss
        return J

    def compute_gradient(self, image, pred, gt):
        image = image.reshape(image.shape[0], -1)
        W_grad = np.dot(image.T, pred-gt)/image.shape[0]
        self.W -= W_grad*self.lr

        b_grad = np.sum(pred-gt)/image.shape[0]
        self.b -= b_grad*self.lr

def train(model):
    x_train, y_train, x_test, y_test = get_data()
    batch_size=250# Change if you want
    epochs=250
    ls_train=[]
    ls_test=[]
    for i in range(epochs):
        loss = []
        for j in range(0,x_train.shape[0], batch_size):
            _x_train = x_train[j:j+batch_size]
            _y_train = y_train[j:j+batch_size]
            out = model.forward(_x_train)
            loss.append(model.compute_loss(out, _y_train))
            model.compute_gradient(_x_train, out, _y_train)
        out = model.forward(x_test)                
        loss_test = model.compute_loss(out, y_test)
        ls_train.append(np.array(loss).mean())
        ls_test.append(loss_test)
        print('Epoch {:6d}: {:.5f} | test: {:.5f}'.format(i, np.array(loss).mean(), loss_test))
    plot(ls_train,ls_test)
    results_model = open('model_emotions.obj', 'wb') 
    pickle.dump(model, results_model)
    results_model.close
                             

def plot(trainloss,testloss):
    fig=plt.figure()
    plt.plot(np.arange(len(trainloss)), trainloss, color='cyan', linewidth=2, label="Train Loss")
    plt.plot(np.arange(len(trainloss)), testloss, color='magenta', linewidth=2, label="Test Loss")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    fig.savefig('Train and Test losses - Emotions.pdf', dpi=fig.dpi)
    plt.close()
    # Save a pdf figure with train and test losses
    #pass

def test(model):
    _, _, x_score, y_score = get_data()
    y_pred=[]
    for im in x_score:
        im = im.reshape(2304)
        pred = np.dot(im, model.W) + model.b
        predicted_class = np.argmax(pred)
        y_pred.append(predicted_class)
    y_score=y_score.astype(int)
    y_pred=np.asarray(y_pred).astype(int)
    ACA = accuracy_score(y_score, y_pred, normalize=True)
    print('ACA: ' + str(ACA))

def demo(model):
    path = './images/'
    random_filename = './images/'+random.choice(os.listdir(path))
    print(random_filename)
    img = Image.open(random_filename).convert('L')
    img = img.resize((48, 48), Image.ANTIALIAS)
    img=np.asarray(img)
    img = img.reshape(2304)
    pred = np.dot(img, model.W) + model.b
    pred=np.argmax(pred)
    label=''
    if pred==0:
        label='Angry'
    elif pred==1:
        label='Disgust'
    elif pred==2:
        label='Happy'
    elif pred==3:
        label='Happy'
    elif pred==4:
        label='Sad'
    elif pred==5:
        label='Surprise'
    elif pred==6:
        label='Neutral'
    img=mpimg.imread(random_filename)
    imgplot = plt.imshow(img)
    plt.title(label)
    plt.show()

if __name__ == '__main__':
    print('Starting...')
    testing=False
    demoing=False

    for opt, arg in opts:
        if opt in ('-t', '--test'):
            testing=True
            print('Test')
        elif opt in ('-d', '--demo'):
            demoing=True
            print('Demo')
            
    if testing:
        if os.path.isfile('./model_emotions.obj'):
            model = pickle.load( open( "model_emotions.obj", "rb" ) )
            test(model)
        else:
            print('Perform training first')
        
    elif demoing:
        if os.path.isfile('./model_emotions.obj'):
            model = pickle.load( open( "model_emotions.obj", "rb" ) )
            demo(model)
        else:
            print('Perform training first')
    else:
        print('Training new model...')
        model = Model()
        train(model)
        test(model)
        
    
