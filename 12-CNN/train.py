import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.measure import compare_ssim as ssim
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_data():

    

    data = np.genfromtxt("list_attr_celeba.csv", dtype=float, delimiter=',', names=True)
    print(data.shape)

    train_labels=[]
    test_labels=[]
    train_paths=[]
    test_paths=[]

    n=0
    for row in data:
        if n<10000:
            labels=np.array([int(row['Eyeglasses']),
                int(row['Bangs']),
                int(row['Black_Hair']),
                int(row['Blond_Hair']),
                int(row['Brown_Hair']),
                int(row['Gray_Hair']),
                int(row['Male']),
                int(row['Pale_Skin']),
                int(row['Smiling']),
                int(row['Young'])])
            train_labels.append(labels)
            train_paths.append(str("{:06d}".format(n+1))+'.jpg')
        elif n<12000:
            labels=np.array([int(row['Eyeglasses']),
                int(row['Bangs']),
                int(row['Black_Hair']),
                int(row['Blond_Hair']),
                int(row['Brown_Hair']),
                int(row['Gray_Hair']),
                int(row['Male']),
                int(row['Pale_Skin']),
                int(row['Smiling']),
                int(row['Young'])])
            test_labels.append(labels)
            test_paths.append(str("{:06d}".format(n+1))+'.jpg')
        else:
            break
        n=n+1


    x_train=[]

    y_train=train_labels
    
    for row in train_paths:
        img = Image.open('./img_align_celeba/'+row)
        img=np.array(img.resize((48,48)))
        x_train.append(img)
            
            
    
    x_train = torch.from_numpy(np.array(x_train)/255)
    y_train = torch.from_numpy(np.array(y_train))

##    print(x_train.size())
##    print(y_train.size())

##    pickle.dump( x_train, open( "pic_tensor.p", "wb" ) )
##    pickle.dump( x_train, open( "pic_tensor.p", "wb" ) )



    return x_train.to(device, dtype=torch.float), y_train.to(device, dtype=torch.float)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,96,5)
        self.conv2 = nn.Conv2d(96,96, 4)
        self.conv3 = nn.Conv2d(96,288, 3)
        self.fc1 = nn.Linear(288*3*3, 700) #Add 1 fully connected layer with 1 neuron
        self.fc2 = nn.Linear(700, 10) #Add 1 fully connected layer with 10 neurons

    def forward(self, x):
        x=x.permute(0, 3, 1,2)
        #print(x.shape)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2) #Perform a Maximum pooling operation over the nonlinear responses of the convolutional layer
        #print(x.shape)
        
        #x = F.dropout(x, 0.25, training=self.training)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        #print(x.shape)
        #x = F.dropout(x, 0.25, training=self.training)
        x = x.view(x.size(0), 288*3*3)
        x = F.relu(self.fc1(x))
        #x = self.fc1(x)
        
        x = self.fc2(x)
        #x=F.softmax(x)
        return x


def train(model):
    x_train, y_train = get_data()
    batch_size = 32 # Change if you want
    epochs = 20 # Change if you want
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)

    for i in range(epochs):
        LOSS = []
        for j in range(0,x_train.shape[0], batch_size):
            _x_train = x_train[j:j+batch_size]
            _y_train = y_train[j:j+batch_size]
            #_y_train = _y_train.long()
            optimizer.zero_grad()
            out = model(_x_train)
            #print(out.shape)
            #print(_y_train.shape)
            loss = criterion(out, _y_train)
            loss.backward()
            optimizer.step()
            LOSS.append(loss.item())
        print('Epoch {:6d}: {:.5f}'.format(i, torch.FloatTensor(LOSS).mean()))
    torch.save(model.state_dict(), 'neural.pth')
if __name__ == '__main__':
    model = Net()
    print(model)
    #model.cuda()
    model.to(device) 
    train(model)
