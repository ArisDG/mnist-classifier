from datetime import datetime
from pandas import read_csv
from pathlib import Path
from shutil import copy
import matplotlib.pyplot as plt
import torch

# Train on GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Define output size of network and layers' size
latent_size = 10

# Leaky ReLU tendency
tendency = 0.2

# Network definition
class Classifier(torch.nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.encode = torch.nn.Sequential(
            torch.nn.Linear(784, int((latent_size+784)/2), bias=True),
            torch.nn.LeakyReLU(tendency),
            torch.nn.Linear(int((latent_size+784)/2), int((3*latent_size+784)/4), bias=True),
            torch.nn.LeakyReLU(tendency),
            torch.nn.Linear(int((3*latent_size+784)/4), latent_size, bias=True),
            torch.nn.Softmax(dim=1)
        )

    def encoder(self, x):
        return self.encode(x)

    def forward(self, input):
        return self.encoder(input)
    
###############################################################################

# Each run creates a new results folder
results = "./classifier_"+datetime.now().strftime("%d-%m-%Y %H:%M:%S")

Path(results+"/loss_curves").mkdir(parents=True, exist_ok=True)
Path(results+"/best_net").mkdir(parents=True, exist_ok=True)

# Save the code that's been used to each run for the sake of mental health
copy('./classifier.py',results)

# Read and preprocess MNIST Dataset
data_train = read_csv('./mnist_train.csv')
data_test = read_csv('./mnist_test.csv')

data_train = data_train.to_numpy()
labels_train = data_train[:,0]
data_train = data_train[:,1:]
data_train = data_train/255

data_test = data_test.to_numpy()
labels_test = data_test[:,0]
data_test = data_test[:,1:]
data_test = data_test/255

# Custom dataset class
class myDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.data[idx]
        labels = self.labels[idx]
        sample = {'data' : torch.tensor(data).float(),'labels' : torch.tensor(labels).float()}
        return sample

# Define batch size
batch_size = 250

# Create Datasets & Dataloaders
train_dataset = myDataset(data_train,labels_train)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers = 0)

test_dataset = myDataset(data_test,labels_test)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle=False, num_workers = 0)


# Init network and optimizer
net = Classifier().double().to(device)

# Define learning rate & initialize optimizer
lr = 0.001

optimizer = torch.optim.Adam(net.parameters(), lr=lr)

# Define the loss function
loss_function = torch.nn.CrossEntropyLoss()

# Loss curve lists to store and print
train_curve_total_loss = [] 
test_curve_total_loss = [] 
train_curve_accuracy = []
test_curve_accuracy = []

print('',end='\r')

for epoch in range(40):

############ TRAIN ###########################################################
    
    # Zero the train epoch loss variables
    train_epoch_total_loss = 0
    test_epoch_total_loss = 0

    train_accuracy = 0
    test_accuracy = 0
    
    for train_batch_index, batch in enumerate(train_dataloader):
        print('\r{}'.format(train_batch_index), end = '\r')

        optimizer.zero_grad()

        # Read a batch of train data
        train_data_batch = batch['data'].double().to(device)
        train_labels_batch = batch['labels'].to(device)

        # Calculate net output for the train batch
        output = net(train_data_batch)
        
        batch_loss = loss_function(output,train_labels_batch.long())

        batch_loss.backward()
        optimizer.step()
        
        train_epoch_total_loss += batch_loss.item()

        for i, label in enumerate(train_labels_batch.detach().cpu().numpy()):
            
            if label == output[i].argmax().detach().cpu().numpy():
                train_accuracy +=1

############ TEST ###########################################################            

    net.eval()

    for test_batch_index, batch in enumerate(test_dataloader):
        print('\r{}'.format(test_batch_index), end = '\r')

        # Read a batch of test data
        test_data_batch = batch['data'].double().to(device)
        test_labels_batch = batch['labels'].to(device)

        # Calculate net output for the test batch
        output = net(test_data_batch)
        
        batch_loss = loss_function(output,test_labels_batch.long())
        
        test_epoch_total_loss += batch_loss.item()

        for i, label in enumerate(test_labels_batch.detach().cpu().numpy()):
            
            if label == output[i].argmax().detach().cpu().numpy():
                test_accuracy +=1

    # Save the latest network    
    torch.save(net.state_dict(), results+'/best_net/latest_net.pth')

    if epoch == 20:
        
        # Define learning rate & initialize optimizer
        lr = 0.00001
        
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        
    net.train()
    
    print('Epoch:', epoch)
    print('Train Loss:',train_epoch_total_loss/((train_batch_index+1)*batch_size))
    print('Test Loss:',test_epoch_total_loss/((test_batch_index+1)*batch_size))

    print('Train Accuracy:',train_accuracy/((train_batch_index+1)*batch_size)*100)
    print('Test Accuracy:',test_accuracy/((test_batch_index+1)*batch_size)*100)

    train_curve_total_loss.append(train_epoch_total_loss/((train_batch_index+1)*batch_size))
    train_curve_accuracy.append(train_accuracy/((train_batch_index+1)*batch_size)*100)

    test_curve_total_loss.append(test_epoch_total_loss/((test_batch_index+1)*batch_size))
    test_curve_accuracy.append(test_accuracy/((test_batch_index+1)*batch_size)*100)

    print('\n')

    plt.figure()
    plt.plot(train_curve_total_loss,label='Train')
    plt.plot(test_curve_total_loss,label='Test')
    plt.title('Loss Curves')
    plt.legend()
    plt.savefig(results+'/loss_curves/loss_curves.png')
    plt.close()

    plt.figure()    
    plt.plot(train_curve_accuracy, label='Train')
    plt.plot(test_curve_accuracy, label='Test')
    plt.title('Accuracy Curves')
    plt.legend()
    plt.savefig(results+'/loss_curves/accuracy_curves.png')
    plt.close()
    
