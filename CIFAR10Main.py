"""
CNN to classify CIFAR-10 images

Created on Sun Oct 11 11:36:32 2020

@author: ancarey
"""
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import numpy as np
import sys 
from CIFAR10Network import ConvNet 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''Show an image'''
def imshow(img):
    img = img / 2 + 0.5 #un-normalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

'''Load and Normalize CIFAR10'''
transform = transforms.ToTensor()

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
validset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

valid_size=0.2
shuffle=True
random_seed=7
batch_size=16
num_train = len(trainset)
indices = list(range(num_train))
split = int(np.floor(valid_size * num_train))

if shuffle:
    np.random.seed(random_seed)
    np.random.shuffle(indices)

train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
    
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=train_sampler)
valloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, sampler=valid_sampler)
testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

dataiter = iter(trainloader)
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

'''Initialize Network, loss function, and optimizer'''
# network = ConvNet()
# network.to(device)

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(network.parameters(), lr=.001, momentum=0.9, nesterov=True, weight_decay=1e-6)
# #optimizer = optim.Adam(network.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-6)
# train_loss = []
# val_loss = []

# '''Train the network'''
# #Need to loop over our data iterator and feed the inputs to the network and optimize
# for epoch in range(80):
#     run_train_loss = 0.0
#     run_val_loss = 0.0
#     total_train_loss = []
#     total_val_loss =[]

#     #Train
#     network.train()
#     for i, data in enumerate(trainloader, 0):
#         inputs, labels = data[0].to(device), data[1].to(device)
        
#         optimizer.zero_grad()
        
#         outputs = network(inputs)
#         tloss = criterion(outputs, labels)
#         tloss.backward()
#         optimizer.step()
#         total_train_loss.append(tloss.item())
#         run_train_loss += tloss.item()
        
#         if i % 100 == 99:    # print every 100 mini-batches
#             print('[%d, %5d] Train loss: %.3f' %
#                   (epoch + 1, i + 1, run_train_loss / 100))
#             run_train_loss = 0.0
    
#     train_loss.append(np.mean(total_train_loss))
    
#     #Evaluate 
#     network.eval()
#     for i, data in enumerate(valloader, 0):
#         vinputs, vlabels = data[0].to(device), data[1].to(device)
        
#         voutputs = network(vinputs)
#         vloss = criterion(voutputs, vlabels)
#         total_val_loss.append(vloss.item())
#         run_val_loss += vloss.item()
        
#         if i % 100 == 99:    # print every 100 mini-batches
#             print('[%d, %5d] Validation loss: %.3f' %
#                   (epoch + 1, i + 1, run_val_loss / 100))
#             run_val_loss = 0.0
            
#     val_loss.append(np.mean(total_val_loss))
    
# plt.plot(train_loss, 'g-', label="train")
# plt.plot(val_loss, 'c-', label="validation")

# plt.xlabel("Epoch")
# plt.ylabel("loss")
# plt.legend()
# plt.title("Training and Validation")
# plt.show()
        
# print('Finished training.')

# '''Save the trained model'''
PATH = './cifar_net.pth'
# torch.save(network.state_dict(), PATH)

'''For testing purposes faster to load the trained model'''
network = ConvNet()
network.to(device) 
network.load_state_dict(torch.load(PATH))

to_print_output_actual = []
to_print_output_pred = []
to_print_ID = []
to_print_ID.append("Image Number")
to_print_output_actual.append("Actual")
to_print_output_pred.append("Predicted")
'''Test the network on the test data'''
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = network(images)
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == labels).squeeze()
        for x in range(len(labels)):
            to_print_output_actual.append(labels.cpu().numpy()[x])
        for y in range(len(predicted)):
            to_print_output_pred.append(predicted.cpu().numpy()[y])
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

h = np.zeros(10)
c = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
    print(100 * class_correct[i] / class_total[i])
    h[i] = float(100 * class_correct[i] / class_total[i])

plt.bar(classes, h)
plt.title("Accuracy Per Class")

for z in range(len(to_print_output_actual) - 1):
    to_print_ID.append(z + 1)

output = np.column_stack((to_print_ID, to_print_output_actual))
output = np.column_stack((output, to_print_output_pred))
np.set_printoptions(threshold=sys.maxsize)
print(output)
file = open("output.txt", "w+")
context = str(output)
file.write(context)
file.close()