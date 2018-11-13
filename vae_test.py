import  torch
import  numpy as np
import  torch.nn.functional as F
import  torchvision
from    torchvision import transforms
import  torch.optim as optim
from    torch import nn




class Encoder(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Encoder, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return F.relu(self.linear2(x))


class Decoder(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Decoder, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return F.relu(self.linear2(x))


class VAE(torch.nn.Module):
    latent_dim = 8

    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self._enc_mu = torch.nn.Linear(100, 8)
        self._enc_log_sigma = torch.nn.Linear(100, 8)

    def _sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = self._enc_mu(h_enc)
        log_sigma = self._enc_log_sigma(h_enc)
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()

        self.z_mean = mu
        self.z_sigma = sigma

        return mu + sigma * std_z.to(h_enc.device).detach()  # Reparameterization trick

    def forward(self, state):
        h_enc = self.encoder(state)
        z = self._sample_latent(h_enc)
        return self.decoder(z)


def latent_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)


if __name__ == '__main__':
    import visdom

    input_dim = 28 * 28
    batch_size = 32
    device = torch.device('cuda')

    transform = transforms.Compose(
        [transforms.ToTensor()])
    mnist = torchvision.datasets.MNIST('db/mnist/', download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(mnist, batch_size=batch_size,
                                             shuffle=True, num_workers=2)

    vis = visdom.Visdom()

    encoder = Encoder(input_dim, 100, 100).to(device)
    decoder = Decoder(8, 100, input_dim).to(device)
    vae = VAE(encoder, decoder).to(device)

    criterion = nn.MSELoss().to(device)

    optimizer = optim.Adam(vae.parameters(), lr=0.0001)
    for epoch in range(100):
        for i, data in enumerate(dataloader, 0):

            inputs, classes = data
            inputs, classes = inputs.resize_(batch_size, input_dim), classes
            inputs, classes = inputs.to(device), classes.to(device)

            optimizer.zero_grad()
            dec = vae(inputs)
            ll = latent_loss(vae.z_mean, vae.z_sigma)
            loss = criterion(dec, inputs) + ll

            loss.backward()
            optimizer.step()
            l = loss.item()
        print(epoch, loss.item())

        vis.images(vae(inputs).cpu().detach().numpy().reshape(32, 1, 28, 28),
                   win='vis_test_x_',
                   nrow=8,
                   opts=dict(title='x_ epoch:%d'%epoch))
        vis.images(inputs.cpu().detach().numpy().reshape(32, 1, 28, 28),
                   win='vis_test_x',
                   nrow=8,
                   opts=dict(title='x epoch:%d'%epoch))

        # plt.imshow(vae(inputs)[0].cpu().detach().numpy().reshape(28, 28), cmap='gray')
        # plt.show(block=True)
