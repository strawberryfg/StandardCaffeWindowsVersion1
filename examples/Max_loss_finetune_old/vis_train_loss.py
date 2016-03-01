import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt

max_show = 0.13

f = open('log/INFO2016-02-26T12-45-09.txt', 'r')
train_loss = []
val_loss = []
for line in f:
    if 'Train net output #1: libxyzloss = ' in line:
        st = line[line.find(' (* 1 = ') + 7:]
        res = st[:st.find(' loss)')]
        train_loss.append(float(res))
    if 'Test net output #0: libxyzloss = ' in line:
        st = line[line.find(' (* 1 = ') + 7:]
        res = st[:st.find(' loss)')]
        #print res
        for i in range(5):
            val_loss.append(float(res))        
f.close()

f = open('log/INFO2016-02-26T12-45-09.txt', 'r')
train_loss_no_cst = []
val_loss_no_cst = []
for line in f:
    if 'Train net output #1: libxyzloss = ' in line:
        st = line[line.find(' (* 1 = ') + 7:]
        res = st[:st.find(' loss)')]
        train_loss_no_cst.append(float(res))
    if 'Test net output #0: libxyzloss = ' in line:
        st = line[line.find(' (* 1 = ') + 7:]
        res = st[:st.find(' loss)')]
        #print res
        for i in range(5):
            val_loss_no_cst.append(float(res))        
f.close()


train_loss = np.asarray(train_loss)
for i in range(len(train_loss)):
    train_loss[i] = max_show if train_loss[i] > max_show else train_loss[i]

val_loss = np.asarray(val_loss)
for i in range(len(val_loss)):
    val_loss[i] = max_show if val_loss[i] > max_show else val_loss[i]


train_loss_no_cst = np.asarray(train_loss_no_cst)
for i in range(len(train_loss_no_cst)):
    train_loss_no_cst[i] = max_show if train_loss_no_cst[i] > max_show else train_loss_no_cst[i]

val_loss_no_cst = np.asarray(val_loss_no_cst)
for i in range(len(val_loss_no_cst)):
    val_loss_no_cst[i] = max_show if val_loss_no_cst[i] > max_show else val_loss_no_cst[i]


fig_loss = plt.figure()
ax_loss = fig_loss.add_subplot(111)
ax_loss.plot(train_loss ,label='train_all', c='b')
ax_loss.plot(val_loss ,label='test_all', c='r')
ax_loss.plot(train_loss_no_cst ,label='train_all', c='k')
ax_loss.plot(val_loss_no_cst ,label='test_all', c='g')
ax_loss.grid()
fig_loss.savefig('train_loss.png')
plt.show()
