import ai_full as ai
import numpy as np

batch_size = 1
latent_size = 100


G_graph = ai.ComputationalGraph()
D_graph = ai.ComputationalGraph()


class Generator(ai.Model):
    def __init__(self):
        self.fc1 = ai.Linear(100, 3 * 3 * 384)
        self.convt1 = ai.ConvTranspose2d(384, 192, kernel_size=5, stride=1, padding=0)
        self.g_bn1 = ai.BatchNorm((192, 7, 7))
        self.convt2 = ai.ConvTranspose2d(192, 96, kernel_size=5, stride=2, padding=2, a=1)
        self.g_bn2 = ai.BatchNorm((96, 14, 14))
        self.convt3 = ai.ConvTranspose2d(96, 1, kernel_size=5, stride=2, padding=2, a=1)
        self.layers = [self.fc1, self.convt1, self.g_bn1, self.convt2, self.g_bn2, self.convt3]

    def forward(self, z):

        o1 = ai.G_graph.relu(self.fc1(z))
        o1 = ai.G_graph.reshape(o1, (384, 3, 3))

        o2 = ai.G_graph.relu(self.convt1(o1))
        o2 = self.g_bn1(o2)

        o3 = ai.G_graph.relu(self.convt2(o2))
        o3 = self.g_bn2(o3)

        o4 = ai.G_graph.tanh(self.convt3(o3))

        return o4

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

        self.layers = [self.d_conv1, self.d_conv2, self.d_bn1, self.d_conv3, self.d_bn2,
                        self.d_conv4, self.d_bn3, self.d_fc]

    def forward(self, image):

        o1 = ai.D_graph.lrelu(self.d_conv1(image))
        o2 = ai.D_graph.lrelu(self.d_bn1(self.d_conv2(o1)))
        o3 = ai.D_graph.lrelu(self.d_bn2(self.d_conv3(o2)))
        o4 = ai.D_graph.lrelu(self.d_bn3(self.d_conv4(o3)))
        o5 = self.d_fc(o4)

        return ai.D_graph.sigmoid(o5)
