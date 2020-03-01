import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.linalg import svd

def get_image_feature(s, k):
	# 对于S，只保留前K个特征值
	s_temp = np.zeros(s.shape[0])
	s_temp[0:k] = s[0:k]
	s = s_temp * np.identity(s.shape[0])
	# 用新的s_temp，以及p,q重构A
	temp = np.dot(p,s)
	temp = np.dot(temp,q)
	plt.imshow(temp, cmap=plt.cm.gray, interpolation='nearest')
	plt.show()

pic = cv2.imread('pics/lena.jpeg')
A = np.array(pic)[:,:,0]
plt.imshow(A, cmap=plt.cm.gray, interpolation='nearest')
plt.show()
# 对图像矩阵A进行奇异值分解，得到p,s,q
p,s,q = svd(A, full_matrices=False)
# 取前k个特征，对图像进行还原
num = s.shape[0]
print("图片特征个数：{}".format(num))
get_image_feature(s, int(num * 0.01))
get_image_feature(s, int(num * 0.1))
get_image_feature(s, int(num * 0.2))
get_image_feature(s, int(num * 0.5))