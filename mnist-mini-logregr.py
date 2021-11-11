import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# importing MNISTmini.mat
mat = scipy.io.loadmat('MNISTmini.mat')

data_fea = mat['train_fea1'].astype(np.float32)/255.0
data_gnd = mat['train_gnd1']

# class 0 = 5s, class 1 = 9s
label = [5, 9]

# Fetch all indices of both classes
c1_idx = np.where(data_gnd[:, 0] == label[0])
c2_idx = np.where(data_gnd[:, 0] == label[1])

# Get 1500 from each class
c1_idx = np.array(c1_idx)
c1_idx = c1_idx[0, 0:1500]
c2_idx = np.array(c2_idx)
c2_idx = c2_idx[0, 0:1500]

# train/val/test split
train_idx = np.concatenate([c1_idx[:500], c2_idx[:500]])  # all_idx[0:1000]
validation_idx = np.concatenate(
    [c1_idx[500:1000], c2_idx[500:1000]]
)  # all_idx[1001:2000]
test_idx = np.concatenate(
    [c1_idx[1000:1500], c2_idx[1000:1500]])  # all_idx[2001:3000]

# x_train: digits, y_train: labels
x_train = data_fea[train_idx, :]
y_train = data_gnd[train_idx, :]

# logregr model initialization and training
plotC = []
plotval = []
plottrain = []
for i in [10**(-7), 10**(-6), 10**(-5), 10**(-4), 10**(-3), 10**(-2), 0.1, 1, 10, 100, 1000, 10000]:  # spacing powers of 10,
    logisticRegr = LogisticRegression(
        penalty='l2', solver='liblinear', C=i, tol=10**(-7), max_iter=1000)
    logisticRegr.fit(x_train, y_train.ravel())

    # Make predictions
    y_pred = logisticRegr.predict(data_fea[validation_idx, :])

    # Assess performance
    val_score = logisticRegr.score(
        data_fea[validation_idx, :], data_gnd[validation_idx, :])
    train_score = logisticRegr.score(
        data_fea[train_idx, :], data_gnd[train_idx, :])

    # Push the performace and value of C into list
    plotval.append(val_score)
    plottrain.append(train_score)
    plotC.append(i)

# Find (x,y) of best validation score
maxVal = max(plotval)
maxValidx = plotval.index(maxVal)
maxCidx = plotC[maxValidx]
print(maxCidx)
print(maxVal)

# Plot the Score of both validation and training
fig = plt.figure()
plt.semilogx(plotC, plotval, color="blue", label="Val_Score", marker="o")
plt.semilogx(plotC, plottrain, color="red", label="Train_Score", marker="o")
plt.legend(loc="lower right")
plt.text(maxCidx, maxVal, ' {} , {}'.format(maxCidx, maxVal))
plt.xlabel("C Values Increasing by Powers of 10")
plt.ylabel("Score Values")

# Show Graph
plt.show()
