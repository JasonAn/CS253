# 3. Read in Data.
tr_num, tt_num = 20000, 2000
import mnist
mndata = mnist.MNIST('./mnist_data_files')
tr_ims, tr_labels = mndata.load_training()
tr_ims, tr_labels = tr_ims[0:tr_num], tr_labels[0:tr_num]
tt_ims, tt_labels = mndata.load_testing()
tt_ims, tt_labels = tt_ims[0:tt_num], tt_labels[0:tt_num]

# 4. Logistic Regression via Gradient Descent.
# def plot4(a, b):
# 	enX = [[1] + tr_ims[i] for i in range(len(tr_ims)) if tr_labels[i] in [a, b]]
# 	enT = [1 if tr_labels[i] == a else 0 for i in range(len(tr_labels)) if tr_labels[i] in [a, b]]
# 	n = int(0.9 * len(enX))
# 	trX, trT = enX[0:n], enT[0:n]
# 	eta = 0.001 / 255 / len(trX)
# 	hdX, hdT = enX[n:], enT[n:]
# 
# 	import math, numpy
# 	def Y(w, X):
# 		return [1 / (1 + math.exp(-numpy.array(w).dot(x))) for x in X]
# 	def E(t, y):
# 		return - sum(t * numpy.log(y) + (1 - numpy.array(t)) * numpy.log(1 - numpy.array(y))) / len(t)
# 	def PC(t, y):
# 		return sum(1 if (t[i] == 1 and y[i] >= 0.5) or (t[i] == 0 and y[i] < 0.5) else 0 for i in range(len(t))) / len(t)
# 
# 	fout = open('eta{}{}.csv'.format(a, b), 'w')
# 	print(eta)
# 	print('t, trE, hdE, enE, trPC, hdPC, enPC')
# 	fout.write('eta = {}\n'.format(eta))
# 	fout.write('t, trE, hdE, enE, trPC, hdPC, enPC\n')
# 	w = numpy.zeros(len(trX[0]))
# 	
# 	w_hist = numpy.zeros([4, len(w)])
# 	hdE_hist = numpy.zeros(4)
# 
# 	t = 0
#     
# 	while(t < 50 and not((hdE_hist[-4] < hdE_hist[-3]) and (hdE_hist[-3] < hdE_hist[-2]) and (hdE_hist[-2] < hdE_hist[-1]))):
# 		y = Y(w, trX)
# 		w = w + [eta * sum((trT[n] - y[n]) * trX[n][j] for n in range(len(trX))) for j in range(len(w))]
# 		trY, hdY, enY = Y(w, trX), Y(w, hdX), Y(w, enX)
# 		trE, hdE, enE = E(trT, trY), E(hdT, hdY), E(enT, enY)
# 		trPC, hdPC, enPC = PC(trT, trY), PC(hdT, hdY), PC(enT, enY)
# 		print(t, trE, hdE, enE, trPC, hdPC, enPC)
# 		fout.write('{}, {}, {}, {}\n'.format(t, trE, hdE, enE, trPC, hdPC, enPC))
# 		
# 		w_hist = numpy.append(w_hist,w)
# 		hdE_hist = numpy.append(hdE_hist, [hdE], axis = 0)
# 		t = t + 1
# 	fout.close()


# z = w[1:]
# print(z)
# z = (numpy.array(z) - min(z)) / (max(z) - min(z)) * 255
# print(z)
# # print(mndata.display(z))
#
# plot4(2,3)
# plot4(2,8)


# 5. Regularization

Lambda = 0.01
import numpy, math
a = 2
b = 3

# def plot5-J_L1(a, b):
# def J_L2(t, y, w):
#     return - sum(t * numpy.log(y) + (1 - numpy.array(t)) * numpy.log(1 - numpy.array(y))) / len(t) + Lambda * dot(w, w)
def J_L1(t, y, w):
    return - sum(t * numpy.log(y) + (1 - numpy.array(t)) * numpy.log(1 - numpy.array(y))) / len(t) + Lambda * sum(numpy.absolute(w))

def Y(w, X):
	return [1 / (1 + math.exp(-numpy.array(w).dot(x))) for x in X]
def E(t, y):
	return - sum(t * numpy.log(y) + (1 - numpy.array(t)) * numpy.log(1 - numpy.array(y))) / len(t)
def PC(t, y):
	return sum(1 if (t[i] == 1 and y[i] >= 0.5) or (t[i] == 0 and y[i] < 0.5) else 0 for i in range(len(t))) / len(t)

enX = [[1] + tr_ims[i] for i in range(len(tr_ims)) if tr_labels[i] in [a, b]]
enT = [1 if tr_labels[i] == a else 0 for i in range(len(tr_labels)) if tr_labels[i] in [a, b]]
n = int(0.9 * len(enX))
trX, trT = enX[0:n], enT[0:n]
eta = 0.001 / 255 / len(trX)
hdX, hdT = enX[n:], enT[n:]

w = numpy.zeros(len(trX[0]))

w_hist = numpy.zeros([4, len(w)])
hdJ_hist = numpy.zeros(4)

print(eta)
print(Lambda)
print('t, trPC, hdPC, enPC')

fout = open('eta_J1{}{}.csv'.format(a, b), 'w')
fout.write('eta_L2 = {}\n'.format(eta))
fout.write('t, rPC, hdPC, enPC\n')

t = 0

while(t < 100 and not((hdJ_hist[-4] < hdJ_hist[-3]) and (hdJ_hist[-3] < hdJ_hist[-2]) and (hdJ_hist[-2] < hdJ_hist[-1]))):
    y = Y(w, trX)
    w = w + [eta * sum((trT[n] - y[n]) * trX[n][j] for n in range(len(trX))) for j in range(len(w))] - 2 * Lambda * w

    trY, hdY, enY = Y(w, trX), Y(w, hdX), Y(w, enX)
    hdJ = J_L1(hdT, hdY, w)
    trPC, hdPC, enPC = PC(trT, trY), PC(hdT, hdY), PC(enT, enY)

    w_hist = numpy.append(w_hist,w)
    hdJ_hist = numpy.append(hdJ_hist, [hdJ], axis = 0)
    print(t, trPC, hdPC, enPC)
    fout.write('{}, {}, {}, {}\n'.format(t, trPC, hdPC, enPC))

    t += 1
fout.close()


