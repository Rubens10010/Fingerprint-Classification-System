#!/usr/bin/env python3
# Does not belong to me!

import matplotlib.pyplot as plt
import numpy as np
from optparse import OptionParser
import scipy.misc as misc
import scipy.ndimage as ndimage
from timeit import default_timer as timer

import utils

parser = OptionParser(usage="%prog [options] sourceimage [destinationimage]")

parser.add_option("-i", dest="images", default=0, action="count",
        help="Show intermediate images.")

parser.add_option("-s", "--subdivide", dest="subdivide",
        default=False, action="store_true",
        help="Iterate the image by subdividing areas.")

parser.add_option("-d", "--dry-run", dest="dryrun", default=False, action="store_true",
        help="Do not save the result.")

parser.add_option("-b", "--no-binarization", dest="binarize", default=True, action="store_false",
        help="Use this option to disable the final binarization step")

options, args = parser.parse_args()

if len(args) == 0 or len(args) > 2:
    parser.print_help()
    exit(1)

sourceImage = args[0]
if len(args) == 1:
    destinationImage = args[0]
else:
    destinationImage = args[1]

if __name__ == '__main__':
    np.set_printoptions(
            threshold=np.inf,
            precision=4,
            suppress=True)

    print("Reading image")
    image = ndimage.imread(sourceImage, mode="L").astype("float64")
    #if options.images > 0:
    utils.showImage(image, "original", vmax=255.0)

    print("Finding mask")
    mask = utils.findMask(image,threshold = 0.09)
    #if options.images > 1:
    utils.showImage(mask, "mask")

    print("Estimating orientations")
    start = timer()
    orientations = np.where(mask == 1.0, utils.estimateOrientations(image,w=10,interpolate=False), -1.0)  # 16 size of block to look for orientation
    end = timer()
    #if options.images > 0:
    utils.showOrientations(image, orientations, "orientations", 10)

    ''''if options.binarize:
        print("Binarizing")
        image = np.where(mask == 1.0, utils.binarize(image), 1.0)
        if options.images > 0:
            utils.showImage(image, "binarized")
    '''
    #if options.images > 0:
    plt.show()
    print(end - start)
    if not options.dryrun:
        misc.imsave(destinationImage, image)
