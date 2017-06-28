from skimage.morphology import skeletonize, thin
import matplotlib.pyplot as plt
from skimage import io

image = io.imread("test.bmp")
skeleton = skeletonize(image)
thinned = thin(image)
thinned_partial = thin(image, max_iter=25)

fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True,
                         subplot_kw={'adjustable': 'box-forced'})
ax = axes.ravel()

ax[0].imshow(image, cmap=plt.cm.gray, interpolation='nearest')
ax[0].set_title('original')
ax[0].axis('off')

ax[1].imshow(skeleton, cmap=plt.cm.gray, interpolation='nearest')
ax[1].set_title('skeleton')
ax[1].axis('off')

ax[2].imshow(thinned, cmap=plt.cm.gray, interpolation='nearest')
ax[2].set_title('thinned')
ax[2].axis('off')

ax[3].imshow(thinned_partial, cmap=plt.cm.gray, interpolation='nearest')
ax[3].set_title('partially thinned')
ax[3].axis('off')

fig.tight_layout()
plt.show()
