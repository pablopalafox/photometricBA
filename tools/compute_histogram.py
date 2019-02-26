import numpy as np
import matplotlib.pyplot as plt
import statistics as stat
from numpy import linalg as LA
import matplotlib.mlab as mlab
import math


global_count = 1


def compute_hist(thingy, blockwise=False):

    print("Computing histogram of", thingy)
    vec = np.fromfile('tests/{}.txt'.format(thingy), dtype=float)


    if blockwise:
        vec = group_residuals_into_blocks(vec)


    mean = np.float32(stat.mean(vec))
    stdev = np.float32(stat.stdev(vec))


    ## zero-mean the input
    vec -= mean


    ## Recompute mean and stdev to check
    mean = np.float32(stat.mean(vec))
    stdev = np.float32(stat.stdev(vec))


    print("mean: ", mean, "stdev", stdev)


    ## Create new figure
    plt.figure(global_count)


    ## Compute histogram
    n, bins, patches = plt.hist(x=vec, bins='auto', color='#0044ff',
                                alpha=0.7, rwidth=None, normed=1)


    ## Define axis, labels, etc
    plt.grid(axis='y', alpha=0.5)
    plt.xlabel('{}'.format(thingy))
    plt.ylabel('Frequency')
    if (thingy == "residuals"):
        if blockwise:
            plt.title('Blockwise Residual Distribution w/o Huber Norm')
        else:
            plt.title('Residual Distribution w/o Huber Norm')
    else:
        plt.title('{} Distribution'.format(thingy))
    maxfreq = n.max()
    plt.text(70, maxfreq/5, r'$\mu=%.2f, \sigma=%.2f$'%(mean, stdev))

    #plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)


    ## Draw Gaussian
    mu = mean
    sigma = stdev
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    plt.plot(x, mlab.normpdf(x, mu, sigma), color='red')


    ## Save figure
    if thingy == "residuals" and blockwise:
        plt.savefig("tests/{}".format("blockwise_" + thingy))
    else:
        plt.savefig("tests/{}".format(thingy))


def group_residuals_into_blocks(vec):
    blocks = []
    current_block = []
    for i, v in enumerate(vec, 1):
        current_block.append(v)
        if (i % 8 == 0 and i != 0):
            current_block_np = np.asarray(current_block)
            norm = LA.norm(current_block_np)
            blocks.append(norm)
            ## Mirror norms to the negative axis
            blocks.append((-norm))
            current_block = []
    return blocks


# compute_hist("b")
# global_count += 1
# compute_hist("a")
# global_count += 1
compute_hist("residuals")
global_count += 1
compute_hist("residuals", blockwise=True)