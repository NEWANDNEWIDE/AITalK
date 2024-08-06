import numpy

class Sigmoid:
    def __init__(self):
        self.w1 = numpy.random.normal()
        self.w2 = numpy.random.normal()
        self.w3 = numpy.random.normal()
        self.w4 = numpy.random.normal()
        self.w5 = numpy.random.normal()
        self.w6 = numpy.random.normal()
        self.b1 = numpy.random.normal()
        self.b2 = numpy.random.normal()
        self.b3 = numpy.random.normal()

    def feedforward(self, x):
        h1 = x[0]*self.w1 + x[1]*self.w2 + self.b1
        h1f = self.sigmoid(h1)
        h2 = x[0]*self.w3 + x[1]*self.w4 + self.b2
        h2f = self.sigmoid(h2)
        o1 = h1f*self.w5 + h2f*self.w6 + self.b3
        of = self.sigmoid(o1)
        return h1, h1f, h2, h2f, o1, of

    def sigmoid(self, x):
        return 1 / (1 + numpy.exp(-x))

    def der_sigmoid(self, x):
        return self.sigmoid(x)*(1-self.sigmoid(x))

    def mse_loss(self, y_tr, y_pre):
        return ((y_tr - y_pre)**2).mean()

    def train(self, data, all_y_tr):
        epochs = 1000
        learn_rate = 0.1
        for i in range(epochs):
            for x, y_tr in zip(data, all_y_tr):
                valcell = self.feedforward(x)
                y_pre = valcell[5]
                der_L_y_pre = -2 * (y_tr - y_pre)
                der_y_pre_h1 = self.der_sigmoid(valcell[4]) * self.w5
                der_y_pre_h2 = self.der_sigmoid(valcell[4]) * self.w6
                der_h1_w1 = self.der_sigmoid(valcell[0]) * x[0]
                der_h1_w2 = self.der_sigmoid(valcell[0]) * x[1]
                der_h2_w3 = self.der_sigmoid(valcell[2]) * x[0]
                der_h2_w4 = self.der_sigmoid(valcell[2]) * x[1]
                der_y_pre_w5 = self.der_sigmoid(valcell[4]) * valcell[1]
                der_y_pre_w6 = self.der_sigmoid(valcell[4]) * valcell[3]
                der_y_pre_b3 = self.der_sigmoid(valcell[4])
                der_h1_b1 = self.der_sigmoid(valcell[0])
                der_h2_b2 = self.der_sigmoid(valcell[2])
                # 重新赋予权值和偏置
                self.w1 -= learn_rate * der_L_y_pre * der_y_pre_h1 * der_h1_w1
                self.w2 -= learn_rate * der_L_y_pre * der_y_pre_h1 * der_h1_w2
                self.w3 -= learn_rate * der_L_y_pre * der_y_pre_h2 * der_h2_w3
                self.w4 -= learn_rate * der_L_y_pre * der_y_pre_h2 * der_h2_w4
                self.w5 -= learn_rate * der_L_y_pre * der_y_pre_w5
                self.w6 -= learn_rate * der_L_y_pre * der_y_pre_w6
                self.b1 -= learn_rate * der_L_y_pre * der_y_pre_h1 * der_h1_b1
                self.b2 -= learn_rate * der_L_y_pre * der_y_pre_h2 * der_h2_b2
                self.b3 -= learn_rate * der_L_y_pre * der_y_pre_b3
                # 每10步输出一次当前损失值
                if i % 10 == 0:
                    y_pred = numpy.apply_along_axis(self.simulate, 1, data)
                    loss = self.mse_loss(all_y_tr, y_pred)
                    print(i, loss)

    def simulate(self, x):
        h1 = x[0] * self.w1 + x[1] * self.w2 + self.b1
        h1f = self.sigmoid(h1)
        h2 = x[0] * self.w3 + x[1] * self.w4 + self.b2
        h2f = self.sigmoid(h2)
        o1 = h1f * self.w5 + h2f * self.w6 + self.b3
        of = self.sigmoid(o1)
        return of
