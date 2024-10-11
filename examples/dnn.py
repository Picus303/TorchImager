import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from TorchImager import Window

torch.set_default_device("cuda")


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


def generate_data(n_samples=1000):
    X = np.random.rand(n_samples, 2)
    y = (X[:, 0] > X[:, 1]).astype(float)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).view(-1, 1)



def train(model, criterion, optimizer, X_train, y_train, epochs=100):
    w1 = Window(2, 4, "grayscale", 100, auto_norm=True)
    w2 = Window(4, 1, "grayscale", 100, auto_norm=True)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

        w1.update(model.fc1.weight)
        w2.update(model.fc2.weight)
        
    w1.close()
    w2.close()



def evaluate(model, X_test):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        predicted_classes = (predictions > 0.5).float()
        return predicted_classes


model = SimpleNet()
print(model.fc1.weight.shape, model.fc2.weight.shape)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

X_train, y_train = generate_data(1000)
X_test, y_test = generate_data(100)

train(model, criterion, optimizer, X_train, y_train, epochs=300)

predictions = evaluate(model, X_test)
accuracy = (predictions == y_test).float().mean()
print(f'Accuracy: {accuracy.item():.4f}')
