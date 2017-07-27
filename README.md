# Fingerprint-Classification-System-Through-Modules
Fingerprint classification system using fingerprint orientantion feature vectors obtained after passinf through different modules - Universidad Nacional de San Agustin (Arequipa - 2017).

# Modules
# Data Storage Module
- NIST4 database
# Data Preprocessing Module
- Preprocessing stages:
  * Fingerprint Histogram Equalization
  * Fingerprint Gabor Enhancement
  * Fingerprint Threshold Binarization
  * Fingerprint Thinning
- Final results:
  * NIST4 4000 thinned fingerprint images
# Data Feature Extraction Module
- Two features extracted:
  * Region of Interes extraction through ANN detection algorithm
  * Orientation Map 100 and 400 features extraction algorithm
- Final results:
  * Manually extracted training database for roi block detection.
  * NIST4 4000 roi fingerprint images of 200x200px
  * 2 dat files containing:
        * 4000 NIST4 features vectors of 400 values from roi orientation map.
        * 4000 NIST4 features vectors of 100 values from roi orientation map.
# Data Processing and Results Gathering( Classification and Results  Module )
- Fingerprint classification system:
  * Fingerprint classification through Stacked Sparse Autoencoder using Keras.
  * Fingerprint classification models with different parameters.
- Final results:
  * Results from experiments of classification system
  * Nice classification model of 88% accuracy found during experimentation available and ready to be loaded in keras.
  * Spanish scratch Paper.
