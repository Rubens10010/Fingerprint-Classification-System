import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, skeletonize_3d
from skimage.data import binary_blobs
from skimage import io

# Invert the horse image
#image = invert(data.horse())
data = io.imread("test.bmp")

#data = binary_blobs(200, blob_size_fraction=.2, volume_fraction=.35, seed=1)

skeleton = skeletonize(data)
skeleton3d = skeletonize_3d(data)

fig, axes = plt.subplots(1, 3, figsize=(8, 4), sharex=True, sharey=True,
                         subplot_kw={'adjustable': 'box-forced'})
ax = axes.ravel()

ax[0].imshow(data, cmap=plt.cm.gray, interpolation='nearest')
ax[0].set_title('original')
ax[0].axis('off')

ax[1].imshow(skeleton, cmap=plt.cm.gray, interpolation='nearest')
ax[1].set_title('skeletonize')
ax[1].axis('off')

ax[2].imshow(skeleton3d, cmap=plt.cm.gray, interpolation='nearest')
ax[2].set_title('skeletonize_3d')
ax[2].axis('off')

fig.tight_layout()
plt.show()
