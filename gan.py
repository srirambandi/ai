import ai_full as ai
import numpy as np


G_graph = ai.ComputationalGraph()
D_graph = ai.ComputationalGraph()

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

        self.layers = [self.g_fc, self.g_bn1, self.g_deconv1, self.g_bn2, self.g_deconv2, self.g_bn3,
                        self.g_deconv3, self.g_bn4, self.g_deconv4]

    def forward(self, z):

        o1 = ai.G_graph.reshape(self.g_fc(z), (8*gf_dim, 2, 2))
        o2 = ai.G_graph.relu(self.g_bn1(o1))
        o3 = ai.G_graph.relu(self.g_bn2(self.g_deconv1(o2)))
        o4 = ai.G_graph.relu(self.g_bn3(self.g_deconv2(o3)))
        o5 = ai.G_graph.relu(self.g_bn4(self.g_deconv3(o4)))
        fake_image = ai.G_graph.tanh(self.g_deconv4(o5))

        return fake_image

class Descriminator(ai.Model):
    def __init__(self):

        self.d_conv1 = ai.Conv2d(1, 64, kernel_size=5, stride=2, padding=2, graph=D_graph)
        self.d_conv2 = ai.Conv2d(64, 2*64, kernel_size=5, stride=2, padding=2, graph=D_graph)
        self.d_bn1 = ai.BatchNorm((2*64, 7, 7))
        self.d_conv3 = ai.Conv2d(2*64, 3*64, kernel_size=5, stride=2, padding=2, graph=D_graph)
        self.d_bn2 = ai.BatchNorm((3*64, 4, 4))
        self.d_conv4 = ai.Conv2d(3*64, 4*64, kernel_size=5, stride=2, padding=2, graph=D_graph)
        self.d_bn3 = ai.BatchNorm((4*64, 2, 2))
        self.d_fc = ai.Linear(1024, 1)

        self.layers = [self.d_conv1, selfs.d_conv2, self.d_bn1, self.d_conv3, self.d_bn2,
                        self.d_conv4, self.d_bn3, self.d_fc]

    def forward(self, image):

        o1 = ai.D_graph.lrelu(self.d_conv1(image))
        o2 = ai.D_graph.lrelu(self.d_bn1(self.d_conv2(o1)))
        o3 = ai.D_graph.lrelu(self.d_bn2(self.d_conv3(o2)))
        o4 = ai.D_graph.lrelu(self.d_bn3(self.d_conv4(o3)))
        o5 = self.d_fc(o4)

        return ai.D_graph.sigmoid(o5)


generator = Generator()
desciminator = Descriminator()


L = ai.Loss(loss_fn='BCELoss', graph=D_graph)
g_optim = ai.Optimizer(generator.layers, optim_fn='Adam', lr=alpha, graph=G_graph)
d_optim = ai.Optimizer(desciminator.layers, optim_fn='Adam', lr=alpha, graph=D_graph)


it, epoch = 0, 0
loss = np.inf

m = 8   # batch size
k = 1   # number of desciminator updates per generator update
data = data_generator(m)

while epoch < 1:

    epoch += 1
    it = 0

    for _ in range(k):

        real_images = data.__next__()
        real_labels = np.ones((1, m))

        real_probs = desciminator.forward(real_images)
        d_loss_real = L.loss(real_probs, real_labels)


        G_graph.grad_mode = False

        z = np.random.randn(z_dim, m)
        fake_images = generator.forward(z)
        fake_labels =  np.zeros((1, m))

        fake_probs = desciminator.forward(fake_images)
        d_loss_fake = L.loss(fake_probs, fake_labels)

        G_graph.grad_mode = True


        d_loss = ai.D_graph.add(d_loss_real, d_loss_fake)
        d_loss.dw = 0

        D_graph.backward()
        d_optim.step()
        d_optim.zero_grad()


    z = np.random.randn(z_dim, m)
    fake_images = generator.forward(z)
    fake_labels =  np.ones((1, m))

    fake_probs = desciminator.forward(fake_images)
    g_loss = L.loss(fake_probs, fake_labels)

    D_graph.backward()
    G_graph.backward()
    g_optim.step()
    g_optim.zero_grad()
    d_optim.zero_grad()


    print('\n\n', 'Epoch {} completed. Accuracy: {}'.format(epoch, evaluate()))


model.save()
