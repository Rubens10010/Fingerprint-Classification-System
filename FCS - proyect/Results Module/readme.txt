FINGERPRINT CLASSIFICATION SYSTEMS DEVELOPMENT BY MODULES
--------------------------------------------------------------------------------------------

Fig Ann_error and Ann_training: Show the accuracy and loss of the trained model for roi detection through a ANN.
Ex10 and Ex11: Show experiment 10 and 11 of experiments made for fingerprint classification with SSAE.
	Ex10: 5 classes problem, 64-32-16 layers, 1000 inputs for training, 3000 testing*
	Ex11: 4 classes problem, 64-32-16 layers, 1000 inputs for training, 3000 testing*
fig1 to fig7: Show the different fases of fingerprint preprocessing:
	fig1: NIST4 Database
	fig2: NIST4 files
	fig3: Histogram equalization
	fig4: Gabor enhancement
	fig5: Fingerprint threshold binarization
	fig6: Fingerprint thinning
	fig7: Fingerprint orientation map extraction
Roi images show:
	Roi_map_filter: filter made with help of the ANN for roi detection.
	Roi_img_filtered: roi threshold filter algorithm applied to this image using above filter.
	Roi_region: Roi region obtained using Roi_map finding the pixel with most belongship to Roi.
group1-4: Accuracy and loss of models for classification system with SSAE and regularization
	top: 400 features x input
	bottom: 100 features x input
	left: 5 classes problem
	right: 4 classes problem
group1-4_err: loss of above models
group8976: Accuracy and loss of models for classification system with SSAE without regularization
	top: 400 features x input
	bottom: 100 features x input
	left: 5 classes problem
	right: 4 classes problem
group8976:	loss of above models


* These plots shows the best results so far