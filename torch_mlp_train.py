import torch.optim
from joblib import dump, load
from sklearn.metrics import mean_squared_error
import torch.optim.lr_scheduler
imputer_path = 'notebooks/imputer.joblib'
transformer_path = 'notebooks/transformer.joblib'

print(imputer, transformer)

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler

if True:
    imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    train_data = imputer.fit('notebooks/train_X.npy')
    imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    train_data = transformer.fit(train_data)

    from joblib import dump, load

    dump(imputer, 'imputer.joblib')
    dump(transformer, 'transformer.joblib')
else:
    imputer = load(imputer_path)
    transformer = load(transformer_path)

class MyDataset(Dataset):

    def __init__(self, x_path, y_path):
        super().__init__()
        X = np.load(x_path)
        self.X = X
        self.y = np.load(y_path)[:, None].astype(np.float32)
        #self.X = imputer.transform(self.X).astype(np.float32)


        self.X = imputer.transform(self.X).astype(np.float32)
        #self.y = transformer.transform(imputer.transform(self.y))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLP(nn.Module):
    def __init__(self, in_features, out_features=1):
        super().__init__()
        self.mlp1 = nn.Linear(in_features, 32)
        self.mlp2 = nn.Linear(32, 64)
        self.mlp3 = nn.Linear(64, out_features)

        #self.act = nn.LeakyReLU(negative_slope=0.05)
        self.act = nn.ReLU()
        self.bn1 = nn.GroupNorm(8, 32) # nn.BatchNorm1d(32)
        self.bn2 = nn.GroupNorm(16, 64) # nn.BatchNorm1d(64)
        # self.bn3 = nn.BatchNorm1d(32)

    def forward(self, x):
        out = self.mlp1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.mlp2(out)
        out = self.bn2(out)
        out = self.act(out)

        out = self.mlp3(out)
        return out


train_dataset = MyDataset("notebooks/train_X.npy", "notebooks/train_y.npy")
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = MyDataset("notebooks/test_X.npy", "notebooks/test_y.npy")
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

EPOCHS = 5

model = MLP(24).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)

scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, pct_start=0.01, max_lr=0.0002, steps_per_epoch=True,
                                                total_steps=(len(train_dataloader) + 1) * EPOCHS)

for epoch in range(EPOCHS):
    model.train()
    for step, (x, y) in enumerate(train_dataloader):
        x, y = x.cuda(), y.cuda()
        out = model(x)
        loss = F.mse_loss(out, y) + F.l1_loss(out, y)
        if step % 1000 == 0:
            print(f'Loss at {step} :', loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    losses = []
    torch.save(model, "fe_net.pt")
    print("Saving intermediate model")
    model.eval()
    targets = []
    preds = []
    with torch.no_grad():
        for x, y in test_dataloader:
            x, y = x.cuda(), y.cuda()
            out = model(x)
            loss = F.mse_loss(out, y)
            targets.append(y.cpu().numpy())
            preds.append(out.cpu().numpy())

    targets = np.concatenate(targets, axis=0)
    preds = np.concatenate(preds, axis=0)
    print("SQRT val", mean_squared_error(preds, targets, squared=False))
