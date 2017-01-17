# 3. Read in Data.
import mnist
tr_num, tt_num = 20000, 2000
mndata = mnist.MNIST('./mnist_data_files')
tr_ims, tr_labels = mndata.load_training()
tr_ims, tr_labels = tr_ims[0:tr_num], tr_labels[0:tr_num]
tt_ims, tt_labels = mndata.load_testing()
tt_ims, tt_labels = tt_ims[0:tt_num], tt_labels[0:tt_num]

# 4. Logistic Regression via Gradient Descent.
import math, numpy
'''
def plot4(a, b):
	enX = [[1] + tr_ims[i] for i in range(len(tr_ims)) if tr_labels[i] in [a, b]]
	enT = [1 if tr_labels[i] == a else 0 for i in range(len(tr_labels)) if tr_labels[i] in [a, b]]
	n = int(0.9 * len(enX))
	trX, trT = enX[0:n], enT[0:n]
	eta = 0.001 / 255 / len(trX)
	# eta = eta0 / (1 + t / T)
	hdX, hdT = enX[n:], enT[n:]

	def Y(w, X):
		return [1 / (1 + math.exp(-numpy.array(w).dot(x))) for x in X]
	def E(t, y):
		return - sum(t * numpy.log(y) + (1 - numpy.array(t)) * numpy.log(1 - numpy.array(y))) / len(t)
	def PC(t, y):
		return sum(1 if (t[i] == 1 and y[i] >= 0.5) or (t[i] == 0 and y[i] < 0.5) else 0 for i in range(len(t))) / len(t)

	fout = open('eta{}{}.csv'.format(a, b), 'w')
	print(eta)
	print('t, trE, hdE, enE, trPC, hdPC, enPC')
	fout.write('eta = {}\n'.format(eta))
	fout.write('t, trE, hdE, enE, trPC, hdPC, enPC\n')
	w = numpy.zeros(len(trX[0]))
	for t in range(100):
		y = Y(w, trX)
		w = w + [eta * sum((trT[n] - y[n]) * trX[n][j] for n in range(len(trX))) for j in range(len(w))]
		trY, hdY, enY = Y(w, trX), Y(w, hdX), Y(w, enX)
		trE, hdE, enE = E(trT, trY), E(hdT, hdY), E(enT, enY)
		trPC, hdPC, enPC = PC(trT, trY), PC(hdT, hdY), PC(enT, enY)
		print(t, trE, hdE, enE, trPC, hdPC, enPC)
		fout.write('{}, {}, {}, {}, {}, {}, {}\n'.format(t, trE, hdE, enE, trPC, hdPC, enPC))
	fout.close()

	def weight2image(w):
		z = w[1:]
		print(z)
		z = (numpy.array(z) - min(z)) / (max(z) - min(z)) * 255
		print(z)
		print(mndata.display(z))
	weight2image(w)

plot4(2,3)
plot4(2,8)
'''

# 5. Regularization
'''
w_hist = numpy.zeros([4, len(w)])
hdE_hist = numpy.zeros(4)

t = 0

while(hdE_hist[0] < hdE_hist[1] and hdE_hist[1] < hdE_hist[2] and hdE_hist[2] < hdE_hist[3]):
	y = Y(w, trX)
	w = w + [eta * sum((trT[n] - y[n]) * trX[n][j] for n in range(len(trX))) for j in range(len(w))]
	trE, hdE, enE = E(trT, Y(w, trX)), E(hdT, Y(w, hdX)), E(T, Y(w, X))

    #w_hist = numpy.append(w_hist,w)

    hdE_hist = numpy.append(hdE_hist, [hdE], axis = 0)

    print(t, trE, hdE, enE)
	fout.write('{}, {}, {}, {}\n'.format(t, trE, hdE, enE))
    t = t + 1
fout.close()
Lambda = 0.01

def J_L2(t, y, w):
    return - sum(t * numpy.log(y) + (1 - numpy.array(t)) * numpy.log(1 - numpy.array(y))) / len(t) + Lambda * dot(w, w)

def J_L1(t, y, w):
    return - sum(t * numpy.log(y) + (1 - numpy.array(t)) * numpy.log(1 - numpy.array(y))) / len(t) + Lambda * sum(absolute(w))
'''

# 6. Softmax Regression via Gradient Descent.
def Y6(W, x):
	t = [math.exp(numpy.array(w).dot(x)) for w in W]
	return numpy.array(t) / sum(t)
def E6(t, Y):
	return - sum(math.log(Y[i][t[i]]) for i in range(len(t))) / len(t)
def PC6(t, Y):
	return sum(numpy.argmax(Y[i]) == t[i] for i in range(len(t))) / len(t)

enX = [[1] + im for im in tr_ims]
enT = tr_labels
n = int(0.9 * len(enX))
trX, trT = enX[0:n], enT[0:n]
eta = 0.001 / 255 / len(trX)
hdX, hdT = enX[n:], enT[n:]
W = numpy.zeros((10, len(trX[0])))

fout = open('eta10.csv', 'w')
fout.write('eta = {}\n'.format(eta))
fout.write('t, trE, hdE, enE, trPC, hdPC, enPC\n')

import time
bais = time.time()
for t in range(100):
	print(time.time() - bais, 's')
	Y = [Y6(W, x) for x in trX]
	W = W + [[eta * sum(((trT[n] == k) - Y[n][k]) * trX[n][j] for n in range(len(trX))) for j in range(len(W[k]))] for k in range(len(W))]
	trY, hdY, enY = [Y6(W, x) for x in trX], [Y6(W, x) for x in hdX], [Y6(W, x) for x in enX]
	trE, hdE, enE = E6(trT, trY), E6(hdT, hdY), E6(enT, enY)
	trPC, hdPC, enPC = PC6(trT, trY), PC6(hdT, hdY), PC6(enT, enY)
	print(t, trE, hdE, enE, trPC, hdPC, enPC)
	fout.write('{}, {}, {}, {}, {}, {}, {}\n'.format(t, trE, hdE, enE, trPC, hdPC, enPC))
fout.close()