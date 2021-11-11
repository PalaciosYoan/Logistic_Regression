import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
mat = scipy.io.loadmat('MNISTmini.mat')

data_fea = mat['train_fea1'].astype(np.float32)/255
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

# train_idx_5 = c1_idx[0,0:500]
# train_idx_9 = c2_idx[0,0:500]
# val_idx_5 = c1_idx[0,500:1000]
# val_idx_9 = c2_idx[0,500:1000]
# test_idx_5 = c1_idx[0,1000:1500]
# test_idx_9 = c2_idx[0,1000:1500]

# # Concatenate arrays and perform random permutation
# all_idx = np.concatenate([c1_idx, c2_idx])
# all_idx = np.random.permutation(all_idx)

# # train/val/test split
# train_idx = all_idx[0:1000]
# validation_idx = all_idx[1001:2000]
# test_idx = all_idx[2001:3000]

train_idx = np.concatenate([c1_idx[:500], c2_idx[:500]])  # all_idx[0:1000]
validation_idx = np.concatenate(
    [c1_idx[500:1000], c2_idx[500:1000]]
)  # all_idx[1001:2000]
test_idx = np.concatenate(
    [c1_idx[1000:1500], c2_idx[1000:1500]])  # all_idx[2001:3000]

# x_train: digits, y_train: labels
x_train = data_fea[train_idx, :]  # .astype(np.float32)/255.0
y_train = data_gnd[train_idx, :]

# logregr model initialization and training
plotx = []
ploty1 = []
ploty = []
for i in [10**(-7), 10**(-6), 10**(-5), 10**(-4), 10**(-3), 10**(-2), 0.1, 1, 10, 100, 1000, 10000]:  # spacing powers of 10,
    logisticRegr = LogisticRegression(
        penalty='l2', solver='liblinear', C=i, tol=10**(-7), max_iter=1000)
    logisticRegr.fit(x_train, y_train.ravel())

    # Make predictions
    # predictions = logisticRegr.predict(data_fea[test_idx, :])
    y_pred = logisticRegr.predict(data_fea[validation_idx, :])

    # # Assess performance
    # score = logisticRegr.score(data_fea[test_idx, :], data_gnd[test_idx, :])
    val_score = logisticRegr.score(
        data_fea[validation_idx, :], data_gnd[validation_idx, :])
    # percentval_score = "{:.1%}".format(val_score)
    train_score = logisticRegr.score(
        data_fea[train_idx, :], data_gnd[train_idx, :])
    # percenttrain_score = "{:.1%}".format(train_score)
    # ploty.append(percentval_score)
    # ploty1.append(percenttrain_score)
    ploty.append(val_score)
    ploty1.append(train_score)
    plotx.append(i)
    # print(score)
df = pd.DataFrame()
df["x"] = plotx
df["val_score"] = ploty
df["train_score"] = ploty1
maxVal = df["val_score"]
maxValidx = maxVal.idxmax()
maxTrain = df["train_score"]
maxTrainidx = maxTrain.idxmax()
x = plotx[maxValidx]
maxVal = ploty[maxValidx]

# ax = plt.gca()
# df.plot(kind="line", x="x", y="val_score", color="blue", ax=ax)
# df.plot(kind="line", x="x", y="train_score", color="red", ax=ax)
fig = plt.figure()
plt.semilogx(plotx, ploty, color="blue", label="Val_Score", marker=".")
plt.semilogx(plotx, ploty1, color="red", label="Train_Score", marker=".")
#plt.scatter(x, maxVal, color="black", zorder=1, label="Max Val_Score")
plt.legend(loc="lower right")
plt.text(x, maxVal, ' {} , {}'.format(x, maxVal))
# fig, ax1 = plt.subplots()
# ax1.plot(x, maxVal, "go", label="marker only")

plt.show()
# once we get good model, run test only once with the good model
