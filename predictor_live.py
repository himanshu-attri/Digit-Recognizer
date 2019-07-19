import joblib,cv2
import numpy as np
model = joblib.load("svm_4label_rbf")

import pyscreenshot as ImageGrab
import time

images_folder = "temp/"
fout = open("testing_x","w+")
for i in range (0,100):
	
	
	img = ImageGrab.grab(bbox=(80, 80, 208, 208)) # X1,Y1,X2,Y2
	print("saved....",i)
	img.save(images_folder+"test_orig.png")
	im = cv2.imread(images_folder+"test_orig.png")
	im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	im_gray = cv2.GaussianBlur(im_gray, (15, 15), 0)

	# Threshold the image
	ret, im_th = cv2.threshold(im_gray, 100, 255, cv2.THRESH_BINARY)


	roi = cv2.resize(im_th, (28, 28), interpolation=cv2.INTER_AREA)

	cv2.imwrite(images_folder+"segmented.png", roi)
	

	rows,cols = roi.shape

	print(rows,cols)
	X=[]

	# #Add pixel one-by-one into data Array.
	for i in range(rows):
	    for j in range(cols):
	        k = roi[i,j]
	        if k>100:
	        	k=1
	        else: 
	        	k=0	
	        X.append(k)

	print("predicting .....")
	#scaling = MinMaxScaler(feature_range=(-1, 1)).fit([X])

	#X = scaling.transform([X])	
	fout.write(str(X))
	predictions = model.predict([X])      
	print(predictions)        
	time.sleep(5)

