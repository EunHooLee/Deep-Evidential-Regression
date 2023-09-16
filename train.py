import numpy as np
import matplotlib.pyplot as plt
import torch

from models.NIG import NormalInverseGammaNetwork

def create_data(x_min, x_max, n, train=True):
    x = np.linspace(x_min, x_max, n)
    x = np.expand_dims(x,-1).astype(np.float32)

    sigma = 3 * np.ones_like(x) if train else np.zeros_like(x)
    y = x**3 + np.random.normal(0, sigma).astype(np.float32)
    
    return torch.FloatTensor(x), torch.FloatTensor(y)

def plot_predictions(x_train, y_train, x_test, y_test, y_pred, n_stds=4, kk=0):
    x_test = x_test[:, 0]
    mu, v, alpha, beta = torch.chunk(y_pred, 4, axis=-1)
    mu = mu[:, 0]
    var = np.sqrt(beta / (v * (alpha - 1)))
    var = np.minimum(var, 1e3)[:, 0]  # for visualization

    plt.figure(figsize=(5, 3), dpi=200)
    plt.scatter(x_train, y_train, s=1., c='#463c3c', zorder=0, label="Train")
    plt.plot(x_test, y_test, 'r--', zorder=2, label="True")
    plt.plot(x_test, mu, color='#007cab', zorder=3, label="Pred")
    plt.plot([-4, -4], [-150, 150], 'k--', alpha=0.4, zorder=0)
    plt.plot([+4, +4], [-150, 150], 'k--', alpha=0.4, zorder=0)
    for k in np.linspace(0, n_stds, 4):
        plt.fill_between(
            x_test, (mu - k * var), (mu + k * var),
            alpha=0.3,
            edgecolor=None,
            facecolor='#00aeef',
            linewidth=0,
            zorder=1,
            label="Unc." if k == 0 else None)
    plt.gca().set_ylim(-150, 150)
    plt.gca().set_xlim(-7, 7)
    plt.legend(loc="upper left")
    plt.show()


def main():
    # 학습 데이터 생성
    x_train, y_train = create_data(-4,4,1000)
    x_test, y_test = create_data(-7,7,1000, train=False)
    
    model = NormalInverseGammaNetwork(1,1,100)
    print(model)
    
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=int(5e-3))
    max_epoch = 500

    model.train()
    
    for epoch in range(max_epoch):
        loss_per_epoch = []
        for batch_idx, (x,y) in enumerate(train_data_loader):
            # print(x.shape)
            optimizer.zero_grad()
            evidence = model(x)
            loss = model.get_loss(evidence,y, 0.01)
            loss.backward()
            optimizer.step()
            loss_per_epoch.append(loss.item())
            
        
        print(f'epoch: {epoch}, loss: {np.mean(loss_per_epoch)}')
            
    
    model.eval()
    with torch.no_grad():
        y_pred = model(x_test)
        print(y_pred.shape)
    plot_predictions(x_train, y_train, x_test, y_test, y_pred)

if __name__ == '__main__':
    main()