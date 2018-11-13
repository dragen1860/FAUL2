import  argparse
import  torch
import  torch.utils.data
from    torch import nn, optim
from    torch.nn import functional as F
from    torchvision import datasets, transforms
from    torchvision.utils import save_image
import  visdom


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)




device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('db/mnist', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('db/mnist', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)




class VAE(nn.Module):

    def __init__(self):
        super(VAE, self).__init__()


        self.fc21 = nn.Linear(32*4*4, 20)
        self.fc22 = nn.Linear(32*4*4, 20)

        # [b, imgc, imgsz, imgsz] => [b, h_c, h_d, h_d]
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=0),  # b, 16, 14, 14
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, 5, stride=2, padding=0),  # b, 32, 7, 7
            nn.LeakyReLU(0.2, inplace=True),
        )

        # [b, q_h_d, 1, 1] => [b, imgc, imgsz, imgsz]
        self.decoder = nn.Sequential(
            # Hout=(Hin−1)×stride[0] − 2×padding[0]+kernel_size[0]+output_padding[0]
            nn.ConvTranspose2d(20, 16, 5, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=2, padding=0),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 4, stride=2, padding=0, output_padding=0),  # b, 1, 28, 28
            # TODO: this can be removed? [-1~1]
            # nn.Tanh()
        )

        inp = torch.Tensor(2, 1, 28, 28)
        h = self.encoder(inp)
        print(h.shape)


    def encode(self, x):
        # h1 = F.relu(self.fc1(x))
        h1 = self.encoder(x)

        h1 = h1.view(h1.size(0), -1)

        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        # h3 = F.relu(self.fc3(z))
        # return torch.sigmoid(self.fc4(h3))
        z = z.view(z.size(0), -1, 1, 1)
        x = self.decoder(z)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criteon = nn.BCEWithLogitsLoss(reduction='sum')

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    # BCE = F.binary_cross_entropy_with_logits(recon_x, x, reduction='sum')

    BCE = criteon(recon_x, x)
    # print(recon_x.shape)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE


def train(epoch, vis):

    model.train()
    train_loss = 0

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device) # [128, 1, 28, 28]

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss, loss_reconstruct = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            # print(recon_batch.shape)
            vis.images(torch.sigmoid(recon_batch).view(-1, 1, 28, 28), nrow=16, win='vae_test')

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item() / len(data)), loss_reconstruct.item()/len(data))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def test(epoch, vis):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar)[0].item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                        recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                           'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


if __name__ == "__main__":
    vis = visdom.Visdom(env='vae')

    for epoch in range(1, args.epochs + 1):
        train(epoch, vis)
        test(epoch, vis)
        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')
