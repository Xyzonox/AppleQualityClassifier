import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import csv
import numpy as np

cuda_id = torch.cuda.current_device()
dense = 248
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.lay1 = nn.Linear(7,dense)
        self.lay2 = nn.Linear(dense,dense)
        self.lay3 = nn.Linear(dense,dense)
        self.lay4 = nn.Linear(dense,1)
    def forward(self, x):
        logits = self.lay1(x)
        logits = nn.functional.leaky_relu(self.lay2(logits))
        logits = nn.functional.leaky_relu(self.lay3(logits))
        logits = nn.functional.tanh(self.lay4(logits))
        return logits

class CustomDataset(Dataset):
    def __init__(self, file):
        inter = np.loadtxt(file, delimiter=",", skiprows=1)
        self.data = (torch.from_numpy(inter).float())
        self.data = self.data.to("cuda")
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        features = self.data[:,[1,2,3,4,5,6,7]]
        label = self.data[:,[8]]
        return features, label
    
def main():
    model = NeuralNetwork()
    model.to("cuda")
    dataset = CustomDataset("apple_quality.csv")
    batch_size = 500
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    validloader = 8
    loss_fn = nn.MSELoss() 
    epoch_number = 0
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    EPOCHS = 50
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, data in enumerate(dataloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss))
            running_loss = 0.0
    torch.save(model, "checkpoints.pth")
    print('Finished Training')

if __name__ == "__main__":
    main()