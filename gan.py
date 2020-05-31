import ai
import numpy as np


z_dim = 100
gf_dim = 64
df_dim = 64

def data_generator(m):

    train_dict = np.load('mnist/train.npy', allow_pickle=True)
    test_dict = np.load('mnist/test.npy', allow_pickle=True)
    data = np.concatenate([train_dict.item()['data'], test_dict.item()['data']])

    while True:
        for batch in range(int(data.shape[0] / m)):
            yield data[batch * m:(batch + 1) * m]


class Generator(ai.Model):
    def __init__(self):

        self.g_fc = ai.Linear(z_dim, 8*gf_dim * 2 * 2)
        self.g_bn1 = ai.BatchNorm((8*gf_dim, 2, 2))
        self.g_deconv1 = ai.ConvTranspose2d(8*gf_dim, 4*gf_dim, kernel_size=5, stride=2, padding=2, a=1)
        self.g_bn2 = ai.BatchNorm((4*gf_dim, 4, 4))
        self.g_deconv2 = ai.ConvTranspose2d(4*gf_dim, 2*gf_dim, kernel_size=5, stride=2, padding=2, a=0)
        self.g_bn3 = ai.BatchNorm((2*gf_dim, 7, 7))
        self.g_deconv3 = ai.ConvTranspose2d(2*gf_dim, gf_dim, kernel_size=5, stride=2, padding=2, a=1)
        self.g_bn4 = ai.BatchNorm((gf_dim, 14, 14))
        self.g_deconv4 = ai.ConvTranspose2d(gf_dim, 1, kernel_size=5, stride=2, padding=2, a=1)

    def forward(self, z):

        o1 = ai.G.reshape(self.g_fc(z), (8*gf_dim, 2, 2))
        o2 = ai.G.relu(self.g_bn1(o1))
        o3 = ai.G.relu(self.g_bn2(self.g_deconv1(o2)))
        o4 = ai.G.relu(self.g_bn3(self.g_deconv2(o3)))
        o5 = ai.G.relu(self.g_bn4(self.g_deconv3(o4)))
        fake_image = ai.G.tanh(self.g_deconv4(o5))

        return fake_image

class Descriminator(ai.Model):
    def __init__(self):

        self.d_conv1 = ai.Conv2d(1, 64, kernel_size=5, stride=2, padding=2)
        self.d_conv2 = ai.Conv2d(64, 2*64, kernel_size=5, stride=2, padding=2)
        self.d_bn1 = ai.BatchNorm((2*64, 7, 7))
        self.d_conv3 = ai.Conv2d(2*64, 3*64, kernel_size=5, stride=2, padding=2)
        self.d_bn2 = ai.BatchNorm((3*64, 4, 4))
        self.d_conv4 = ai.Conv2d(3*64, 4*64, kernel_size=5, stride=2, padding=2)
        self.d_bn3 = ai.BatchNorm((4*64, 2, 2))
        self.d_fc = ai.Linear(1024, 1)

    def forward(self, image):

        o1 = ai.G.lrelu(self.d_conv1(image))
        o2 = ai.G.lrelu(self.d_bn1(self.d_conv2(o1)))
        o3 = ai.G.lrelu(self.d_bn2(self.d_conv3(o2)))
        o4 = ai.G.lrelu(self.d_bn3(self.d_conv4(o3)))
        o5 = self.d_fc(o4)

        return ai.G.sigmoid(o5)


generator = Generator()
descriminator = Descriminator()
print(generator)
print(descriminator)


L = ai.Loss(loss_fn='BCELoss')
g_optim = ai.Optimizer(generator.parameters(), optim_fn='Adam', lr=alpha)
d_optim = ai.Optimizer(descriminator.parameters(), optim_fn='Adam', lr=alpha)


it, epoch = 0, 0
m = 8   # batch size
k = 1   # number of descriminator updates per generator update
data = data_generator(m)


def evaluate():
    G.grad_mode = False

    z = np.random.randn(z_dim, m)
    fake_images = generator.forward(z)

    G.grad_mode = True


while epoch < 1:

    epoch += 1
    it = 0

    for p in generator.parameters():
        p.requires_grad = False

    for _ in range(k):

        real_images = data.__next__()
        real_labels = np.ones((1, m))

        real_probs = descriminator.forward(real_images)
        d_loss_real = L.loss(real_probs, real_labels)

        z = np.random.randn(z_dim, m)
        fake_images = generator.forward(z)
        fake_labels =  np.zeros((1, m))

        fake_probs = descriminator.forward(fake_images)
        d_loss_fake = L.loss(fake_probs, fake_labels)

        d_loss = ai.G.add(d_loss_real, d_loss_fake)
        d_loss.grad = 0

        d_loss.backward()
        d_optim.step()
        d_optim.zero_grad()

    for p in generator.parameters():
        p.requires_grad = True

    z = np.random.randn(z_dim, m)
    fake_images = generator.forward(z)
    fake_labels =  np.ones((1, m))

    fake_probs = descriminator.forward(fake_images)
    g_loss = L.loss(fake_probs, fake_labels)

    g_loss.backward()
    g_optim.step()
    g_optim.zero_grad()
    d_optim.zero_grad()


    print('\n\n', 'Epoch {} completed. Accuracy: {}'.format(epoch, evaluate()))


generator.save()
descriminator.save()
