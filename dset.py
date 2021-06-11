import numpy as np

class linear():
    def __init__(self):
        self.numlow, self.numhigh = 30, 6
        self.pick_interval, self.start = int(self.numlow / self.numhigh), 0
        self.x_low = [0]
        self.x_high = [1]
        self.learning_rate_low = 7e-4
        self.learning_rate_high = 5e-4
        self.iteration_low = 20000
        self.iteration_high = 13000
        
    def high_fidelity(self, xh):
        yh = np.power((6*xh - 2), 2) * np.sin(12*xh - 4)
        yh = np.reshape(yh, (yh.shape[0], 1))
        return yh
    
    def low_fidelity(self, xl):
        yh = self.high_fidelity(xl)
        yl = 0.5 * yh + 10 * (xl - 0.5) + 5
        yl = np.reshape(yl, (yl.shape[0], 1))
        return yl

class step():
    def __init__(self):
        self.numlow, self.numhigh = 30, 5
        self.pick_interval, self.start = int(self.numlow / self.numhigh), 4
        self.x_low = [0]
        self.x_high = [2]
        self.learning_rate_low = 1e-3
        self.learning_rate_high = 7e-4
        self.iteration_low = 12000
        self.iteration_high = 12000

    def high_fidelity(self, xh):
        highnoise = np.random.normal(0, 0.01, [xh.shape[0], 1])
        yh = np.where(xh <= 1, -1, 2)
        yh = np.reshape(yh, (yh.shape[0], 1)) + highnoise
        return yh

    def low_fidelity(self, xl):
        lownoise = np.random.normal(0, 0.01, [xl.shape[0], 1])
        yl = np.where(xl <= 1, 0, 1)
        yl = np.reshape(yl, (yl.shape[0], 1)) + lownoise
        return yl


class sine():
    def __init__(self):
        self.numlow, self.numhigh = 30, 10
        self.pick_interval, self.start = int(self.numlow / self.numhigh), 1
        self.x_low = [0]
        self.x_high = [1]
        self.learning_rate_low = 1e-3
        self.learning_rate_high = 1e-3
        self.iteration_low = 30000
        self.iteration_high = 17000

    def high_fidelity(self, xh):
        yh = xh * np.exp(self.low_fidelity(xh) * (2 * xh - .3)) - 1
        yh = np.reshape(yh, (yh.shape[0], 1))
        return yh

    def low_fidelity(self, xl):
        yl = np.cos(15 * xl)
        yl = np.reshape(yl, (yl.shape[0], 1))
        return yl


class currin():
    def __init__(self):
        self.numlow, self.numhigh = 30, 6
#         self.numlow, self.numhigh = 150, 30
        self.pick_interval, self.start = int(self.numlow / self.numhigh), 3
        self.x_low = [0, 0]
        self.x_high = [1, 1]
        self.learning_rate_low = 1e-3
        self.learning_rate_high = 1e-3
        self.iteration_low = 14000
        self.iteration_high = 14000

    def high_fidelity(self, xh):
        f2 = lambda x: (1 - np.exp(-1 / (2 * x[1]))) * \
                       (2300 * np.power(x[0], 3) + 1900 * np.power(x[0], 2) + 2092 * x[0] + 60) / \
                       (100 * np.power(x[0], 3) + 500 * np.power(x[0], 2) + 4 * x[0] + 20)
        f1 = lambda x: 0.25 * f2(x + np.array([0.05, 0.05])) + \
                       0.25 * f2(x + np.array([0.05, max(-0.05, -x[1])])) + \
                       0.25 * f2(x + np.array([-0.05, 0.05])) + \
                       0.25 * f2(x + np.array([-0.05, max(-0.05, -x[1])]))
        yh = np.apply_along_axis(f2, 1, xh)
        yh = np.reshape(yh, [len(yh), 1])
        return yh

    def low_fidelity(self, xl):
        f2 = lambda x: (1 - np.exp(-1 / (2 * x[1]))) * \
                       (2300 * np.power(x[0], 3) + 1900 * np.power(x[0], 2) + 2092 * x[0] + 60) / \
                       (100 * np.power(x[0], 3) + 500 * np.power(x[0], 2) + 4 * x[0] + 20)
        f1 = lambda x: 0.25 * f2(x + np.array([0.05, 0.05])) + \
                       0.25 * f2(x + np.array([0.05, max(-0.05, -x[1])])) + \
                       0.25 * f2(x + np.array([-0.05, 0.05])) + \
                       0.25 * f2(x + np.array([-0.05, max(-0.05, -x[1])]))
        yl = np.apply_along_axis(f1, 1, xl)
        yl = np.reshape(yl, [len(yl), 1])
        return yl


class park():
    def __init__(self):
        self.numlow, self.numhigh = 30, 6
        self.pick_interval, self.start = int(self.numlow / self.numhigh), 3
        self.x_low = [0, 0, 0, 0]
        self.x_high = [1, 1, 1, 1]
        self.learning_rate_low = 1e-3
        self.learning_rate_high = 1e-3
        self.iteration_low = 20000
        self.iteration_high = 20000

    def high_fidelity(self, xh):
        f2 = lambda x: x[0] / 2 * (np.sqrt(1 + (x[1] + np.square(x[2])) * x[3] / np.square(x[0])) - 1) + \
                       (x[0] + 3 * x[3]) * np.exp(1 + np.sin(x[2]))
        f1 = lambda x: (1 + np.sin(x[0]) / 10) * f2(x) - 2 * x[0] + np.square(x[1]) + np.square(x[2]) + 0.5
        yh = np.apply_along_axis(f2, 1, xh)
        yh = np.reshape(yh, [yh.shape[0], 1])
        return yh

    def low_fidelity(self, xl):
        f2 = lambda x: x[0] / 2 * (np.sqrt(1 + (x[1] + np.square(x[2])) * x[3] / np.square(x[0])) - 1) + \
                       (x[0] + 3 * x[3]) * np.exp(1 + np.sin(x[2]))
        f1 = lambda x: (1 + np.sin(x[0]) / 10) * f2(x) - 2 * x[0] + np.square(x[1]) + np.square(x[2]) + 0.5
        yl = np.apply_along_axis(f1, 1, xl)
        yl = np.reshape(yl, [yl.shape[0], 1])
        return yl
    
class park2():
    def __init__(self):
        self.numlow, self.numhigh = 15, 5
        self.pick_interval, self.start = int(self.numlow / self.numhigh), 0
        self.x_low = [0, 0, 0, 0]
        self.x_high = [1, 1, 1, 1]
        self.learning_rate_low = 1e-3
        self.learning_rate_high = 1e-3
        self.iteration_low = 20000
        self.iteration_high = 20000
        
    def high_fidelity(self, x):
        f = lambda x: (2/3) * np.exp(x[0] + x[1]) - x[3]*np.sin(x[2]) + x[2]
        yh = np.apply_along_axis(f, 1, x)
        yh = np.reshape(yh, (yh.shape[0], 1))
        return yh
    
    def low_fidelity(self, x):
        yh = self.high_fidelity(x)
        yl = 1.2 * yh - 1
        yl = np.reshape(yl, (yl.shape[0], 1))
        return yl