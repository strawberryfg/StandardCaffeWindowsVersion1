import sys
sys.path.insert(0, '../../python')
sys.path.insert(0, '/usr/local/lib/python2.7/site-packages')
import caffe
caffe.set_mode_gpu()
net = caffe.Net( 'DeepModel.prototxt',
                'models/_iter_100000.caffemodel',
                caffe.TEST)
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import pylab
from mpl_toolkits.mplot3d import Axes3D
import random
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import *
from numpy import *
from matplotlib import pyplot
import datetime
import cv2
for k, v in net.blobs.items():
    print k, v.data.shape
    
    
NYU_joint = [0, 3, 5, 8, 10, 13, 15, 18, 24, 25, 26, 28, 29, 30]
J = len(NYU_joint)
test_num = 8252
batch_size = (net.blobs['depth'].shape)[0]
iters = test_num / batch_size
error_per_frame = np.zeros(iters * batch_size)
out_avg_thresh = 10000
thresh = np.zeros(100)
joint_error = np.zeros(J)
vis_online = False
output_feature = True

feats = np.zeros((test_num, 55), dtype = 'float32')
gts = np.zeros((test_num, 93), dtype = 'float32')

sum = 0
a = datetime.datetime.now()
for t in range(iters):
    net.forward()
    for i in range(0,batch_size):
        img_id = batch_size * t + i
        ratio = 150 if img_id < 2440 else 130
        
        x = np.zeros(J)
        gtx = np.zeros(J)
        y = np.zeros(J)
        gty = np.zeros(J)
        z = np.zeros(J)
        gtz = np.zeros(J)
        for j in range(0,J):
            x[j]=net.blobs['libxyz'].data[i][NYU_joint[j] * 3]
            gtx[j]=net.blobs['joint_xyz'].data[i][NYU_joint[j] * 3]
            y[j]=net.blobs['libxyz'].data[i][NYU_joint[j] * 3 + 1]
            gty[j]=net.blobs['joint_xyz'].data[i][NYU_joint[j] * 3 + 1]            
            z[j]=net.blobs['libxyz'].data[i][NYU_joint[j] * 3 + 2]
            gtz[j]=net.blobs['joint_xyz'].data[i][NYU_joint[j] * 3 + 2]
            joint_error[j] += ratio * ((x[j] - gtx[j]) * (x[j] - gtx[j]) + (y[j] - gty[j]) * (y[j] - gty[j]) + (z[j] - gtz[j]) * (z[j] - gtz[j])) ** 0.5
        error_max = ((((x - gtx) * (x - gtx) + (y - gty) * (y - gty) + (z - gtz) * (z - gtz)) ** 0.5).max()) * ratio
        error_avg = ((((x - gtx) * (x - gtx) + (y - gty) * (y - gty) + (z - gtz) * (z - gtz)) ** 0.5).sum() / J) * ratio
        for k in range(int(error_max), 100):
            thresh[k] += 1
        sum += error_avg
        error_per_frame[img_id] = error_avg
        if error_avg > out_avg_thresh:
            img = (net.blobs['depth'].data[i][0] + 1) / 2 * 255
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            for l in range(J):
                cv2.circle(img, (int((x[l] + 1) / 2 * 128), int((- y[l] + 1) / 2 * 128)), 2, (255, 0, 0), 2)
                #cv2.circle(img, (int((gtx[i] + 1) / 2 * 128), int((-gty[i] + 1) / 2 * 128)), 2, (0, 0, 255), 2)
                cv2.imwrite('output/{:d}_{:.4f}.png'.format(img_id, error_avg), img)
                #cv2.imwrite('output/{:d}.png'.format(I * BATCHSIZE + see), img)
        if vis_online:
            fig=plt.figure()
            ax=fig.add_subplot((111),projection='3d')
            ax.set_xlabel('z')
            ax.set_ylabel('x')
            ax.set_zlabel('y')
            ax.scatter(z,x,y)
            ax.scatter(gtz, gtx, gty, c = 'r')
            plt.show()
            
        if output_feature:
            feats[img_id] = net.blobs['libdof'].data[i] 
            gts[img_id] = net.blobs['joint_xyz'].data[i]
    print 'iter = ', t, ', current_error = ', sum / (batch_size * (t + 1))


joint_error /= (1.0 * batch_size * iters)    
b = datetime.datetime.now()
c = b - a
tm = 1000. * c.seconds / (batch_size * iters)
print 'average time = ',tm , 'ms ', 'fps = ', 1000. / tm 
print 'average error =',  sum / (batch_size * iters)

if output_feature:
    n = len(feats)
    out = open('ours_feats.txt', 'w')
    for i in range(n):
        for j in range(55):
            out.write('{:.6f} '.format(feats[i, j]))
        out.write('\n')
    out.close()

    out = open('gts.txt', 'w')
    for i in range(n):
        for j in range(93):
            out.write('{:.6f} '.format(gts[i, j]))
        out.write('\n')
    out.close()

fig_thresh = plt.figure()
ax_thresh = fig_thresh.add_subplot(111)
ax_thresh.plot(thresh / (1.0 * batch_size * iters) ,label='test_all', c='b')
ax_thresh.grid()
fig_thresh.savefig('max_error_under_threshold.png')

fig_error_per_frame = plt.figure()
ax_error_per_frame = fig_error_per_frame.add_subplot(111)
ax_error_per_frame.plot(error_per_frame, label='test_all', c='b')
ax_error_per_frame.grid()
fig_error_per_frame.savefig('error_per_frame.png')

NYU_joint_name = ['P1', 'P2', 'R1', 'R2', 'M1', 'M2', 'I1', 'I2', 'C', 'W1', 'W2', 'T1', 'T2', 'T3']
inds = np.arange(len(joint_error))
fig_joint_error, ax_joint_error = plt.subplots()
ax_joint_error.bar(inds, joint_error)
ax_joint_error.set_xticks(inds + 0.5)
ax_joint_error.set_xticklabels(NYU_joint_name)
plt.title("Mean Error Per Joints")
plt.ylabel("Error(mm)")
fig_joint_error.savefig('mean_error_per_joint.png')

import cPickle 
cPickle.dump((thresh / (1.0 * batch_size * iters), joint_error), open('param.pkl', 'w'))

#plt.show()