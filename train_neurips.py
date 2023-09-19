import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader 

from models.NIG import NormalInverseGammaNetwork
from loss import EvidentialRegressionLoss

def get_data_iterator(iterable):
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()

def plot_predictions(x_train, y_train, x_test, y_test, y_pred, n_stds=4, kk=0):
    x_test = x_test[:, 0]
    # mu, v, alpha, beta = np.split(y_pred, 4, axis=-1)
    mu, v, alpha, beta = y_pred
    mu = mu[:, 0]
    var = np.sqrt(beta / (v * (alpha - 1)))
    var = np.minimum(var, 1e3)[:, 0]  # for visualization
    print(var)
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


def create_data(x_min, x_max, n, train=True):
    x = np.linspace(x_min, x_max, n)
    x = np.expand_dims(x, -1).astype(np.float32)

    sigma = 3 * np.ones_like(x) if train else np.zeros_like(x)
    y = x ** 3 + np.random.normal(0, sigma).astype(np.float32)
    return torch.from_numpy(x), torch.from_numpy(y)


def main():
    x_train, y_train = create_data(-4,4,1000)
    x_test, y_test = create_data(-7, 7, 1000, train=False)
    train_dataset = TensorDataset(x_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=0)
    train_iter = get_data_iterator(train_dataloader)


    model = NormalInverseGammaNetwork(1,1,64)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    
    max_iters = 150000
    model.train()
    try:
        it = 0
        while it < max_iters:
            x, y = next(train_iter)
            evidence = model(x)
            loss = EvidentialRegressionLoss(y,evidence,1e-2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if it % 100 == 0:
                print(f'Loss: {loss.item()}')
            it+=1

    except KeyboardInterrupt:
        print('Terminating...')

    model.eval()
    with torch.no_grad():
        y_pred = model(x_test)
    plot_predictions(x_train, y_train, x_test, y_test, y_pred)


if __name__ == '__main__':
    main()