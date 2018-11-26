import pickle
import numpy as np
data=open("/home/ameya/Desktop/DATA ANALYSIS COURSE/RML2016.10a_dict.pkl","rb")
final=pickle.load(data)
#cast=np.array(final)
#print(len(final))
y1 = final.keys()
y = np.array(y1)
x5 = final.values()
x = np.array(x5)
#print(len(y))
#print(len(x))
#print(len(x[1]))
#print(len(x[1][1]))
#print(len(x[1][1][1]))
#print(y[1].shape)
#print(len(y[1][1]))
#print(len(x))
#print(x)
#print(x[1])
#print(x[1][1])
#print(x[1][1][1])
#print(len(x[1]))
xtrain = []
j=0
m = 0
while j < 220:
	while m<800:
	    xtrain.append(x5[j][m])
	    m = m+1
	m = 0
	j = j+1
#print(len(xtrain))
xtest = []
j=0
m = 800
while j < 220:
	while m<1000:
	    xtest.append(x5[j][m])
	    m = m+1
	m = 800
	j = j+1
#print(len(xtest))
ytrain = []
ytest = []
i = 0
j=0
#print('a')
while i<220:
	#print('b')
	if y[i][0]=='8PSK':
		#print('c')
		j = 0
		while j < 800:
			ytrain.append([1,0,0,0,0,0,0,0,0,0,0])
			j = j+1
		#print('d')
		while j < 1000:
			ytest.append([1,0,0,0,0,0,0,0,0,0,0])
			j = j+1
		j = 0
		#print('e')
	elif y[i][0] == 'AM-DSB':
		#print('1')
		while j < 800:
			ytrain.append([0,1,0,0,0,0,0,0,0,0,0])
			j = j+1
		#print('2')
		while j < 1000:
			ytest.append([0,1,0,0,0,0,0,0,0,0,0])
			j = j+1
		j = 0
		#print('3')
	elif y[i][0] == 'AM-SSB':
		#print('a')
		while j < 800:
			ytrain.append([0,0,1,0,0,0,0,0,0,0,0])
			j = j+1
		#print('4')
		while j < 1000:
			ytest.append([0,0,1,0,0,0,0,0,0,0,0])
			j = j+1
		j = 0	
		#print('5')	
	elif y[i][0] == 'BPSK':
		#print('6')
		while j < 800:
			ytrain.append([0,0,0,1,0,0,0,0,0,0,0])
			j = j+1
		#print('7')
		while j < 1000:
			ytest.append([0,0,0,1,0,0,0,0,0,0,0])
			j = j+1
		j = 0
		#print('8')
	elif y[i][0] == 'CPFSK':
		#print('9')
		while j < 800:
			ytrain.append([0,0,0,0,1,0,0,0,0,0,0])
			j = j+1
		#print('10')
		while j < 1000:
			ytest.append([0,0,0,0,1,0,0,0,0,0,0])
			j = j+1
		j = 0
		#print('11')
	elif y[i][0] == 'GFSK':
		#print('a')
		while j < 800:
			ytrain.append([0,0,0,0,0,1,0,0,0,0,0])
			j = j+1
		#print('12')
		while j < 1000:
			ytest.append([0,0,0,0,0,1,0,0,0,0,0])
			j = j+1
		j = 0
		#print('13')
	elif y[i][0] == 'PAM4':
		#print('14')
		while j < 800:
			ytrain.append([0,0,0,0,0,0,1,0,0,0,0])
			j = j+1
		#print('15')
		while j < 1000:
			ytest.append([0,0,0,0,0,0,1,0,0,0,0])
			j = j+1
		j = 0
		#print('16')
	elif y[i][0] == 'QAM16':
		#print('a11')
		while j < 800:
			ytrain.append([0,0,0,0,0,0,0,1,0,0,0])
			j = j+1
		#print('17')
		while j < 1000:
			ytest.append([0,0,0,0,0,0,0,1,0,0,0])
			j = j+1
		j = 0
		#print('18')
	elif y[i][0] == 'QAM64':
		while j < 800:
			ytrain.append([0,0,0,0,0,0,0,0,1,0,0])
			j = j+1
		#print('19')
		while j < 1000:
			ytest.append([0,0,0,0,0,0,0,0,1,0,0])
			j = j+1
		j = 0
		#print('20')
	elif y[i][0] == 'QPSK':
		while j < 800:
			ytrain.append([0,0,0,0,0,0,0,0,0,1,0])
			#print(j)
			j = j+1
			#print(len(ytrain))
		#print('21')
		while j < 1000:
			ytest.append([0,0,0,0,0,0,0,0,0,1,0])
			j = j+1
		j = 0
		#print('22')
	elif y[i][0] ==  'WBFM':
		while j < 800:
			ytrain.append([0,0,0,0,0,0,0,0,0,0,1])
			j = j+1
		#print('23')
		while j < 1000:
			ytest.append([0,0,0,0,0,0,0,0,0,0,1])
			j = j+1
		j = 0
		#print('24')
	#print('25')
	#print(len(ytrain))
	i = i+1
	#print('26')
#print(len(ytrain))
#print(len(ytest))
xtrain = np.array(xtrain)
#print(np.shape(xtrain))
ytrain = np.array(ytrain)
#print(np.shape(ytrain))
#print(z)	
#print(w)
#print(np.shape(x1))
#print(np.shape(z1))
import keras
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, Flatten,Dropout,Activation,Reshape,ZeroPadding2D

#dr = 0.5 # dropout rate (%)
#model = models.Sequential()
# model.add(Reshape([1]+in_shp, input_shape=in_shp))
# model.add(ZeroPadding2D((0, 2)))
# model.add(Convolution2D(256, 1, 3, border_mode='valid', activation="relu", name="conv1", init='glorot_uniform'))
# model.add(Dropout(dr))
# model.add(ZeroPadding2D((0, 2)))
# model.add(Convolution2D(80, 2, 3, border_mode="valid", activation="relu", name="conv2", init='glorot_uniform'))
# model.add(Dropout(dr))
# model.add(Flatten())
# model.add(Dense(256, activation='relu', init='he_normal', name="dense1"))
# model.add(Dropout(dr))
# model.add(Dense( len(classes), init='he_normal', name="dense2" ))
# model.add(Activation('softmax'))
# model.add(Reshape([len(classes)]))
# model.compile(loss='categorical_crossentropy', optimizer='adam')
# model.summary()

model = Sequential()
model.add(Reshape([1]+[2, 128], input_shape=[2, 128]))
model.add(ZeroPadding2D((0, 2)))
model.add(Convolution2D(256, (1, 3), border_mode='valid', activation="relu"))
model.add(Dropout(0.5))
model.add(ZeroPadding2D((0, 2)))
model.add(Convolution2D(80, (1, 3), border_mode="valid", activation="relu"))
model.add(Dropout(0.5))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
model.add(Convolution2D(50, (1, 2), border_mode="valid", activation="relu"))
model.add(Dropout(0.2))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(11))
model.add(Activation('softmax'))
model.add(Reshape([11]))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(xtrain,ytrain,batch_size=1024,epochs=1,validation_split=0.2)

score = model.evaluate(np.array(xtest),np.array(ytest),verbose=0,batch_size=200)
print score
snr=[]
types=[]
j = 0
while j<220:
	if j==0:
		score1 = model.evaluate(np.array(xtest[:200]),np.array(ytest[:200]), verbose=0)
		snr = snr + [list(y1[j]) + list(score1)]
		print score1
	else:
		score1 = model.evaluate(np.array(xtest[200*(j-1):200*j]),np.array(ytest[200*(j-1):200*j]),  verbose=0)
		snr = snr + [list(y1[j]) + list(score1)]
		print score1
	j = j+1

print(snr)

#print(y)
#print(x)
