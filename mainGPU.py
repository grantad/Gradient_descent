import torch
import torch.nn as nn

# Define a simple linear regression model
model = nn.Linear(1,1)

# Define a custom dataset class
class Dataset(torch.untils.data.Dataset):
    def __init__(self):
        self.X = torch.randn(1000, 1)
        self.y = torch.randn(1000, 1)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x, y = self.X[idx], self.y[idx]
        return x, y
    

# Create a data loader
dataset = Dataset()
data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Train the model using GPU
device = torch.device('cuda:0', if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoc in range(100):
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()