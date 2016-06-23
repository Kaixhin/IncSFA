IncSFA
======

Incremental Slow Feature Analysis, ported [from MATLAB](http://people.idsia.ch/~luciw/incsfa.html).

Examples
--------

### `example_low_dim`
This is the toy example, first introduced by Wiskott in the SFA paper. It will show the feature outputs, slowness measured, and feature correlation during the training. IncSFA is driven to learn slow features that are as decorrelated as possible. One can watch this interplay between the two constraints during the learning.

### `example_high_dim_images`
Requires `hdf5`.  
Here, IncSFA is applied to a sequence of high-dimensional images. It uses the iCubArm data in the Data folder (not included in this repo), stored already in matrix form in robotdata.h5.zip. During training, it will show the feature outputs on a few selected episodes, as well as the slowness measured and the mutual correlation of the features. This takes over 100 episodes of training to really start showing good features, sometimes longer. It is currently set to stop at 200 episodes. Once it stops, it will run `view_result`, which shows side-by-side the image sequences and the slow feature embedding. During training, it periodically stores the features in `feature_saved.t7`.

### `view_result`
Used to view the embedding.

Functions
---------

| Function   | Description                                                                                            |
|------------|--------------------------------------------------------------------------------------------------------|
| amnesic    | Amnesic averaging, to set learning rates.                                                              |
| quadexpand | Quadratic expansion of a column vector.                                                                |
| CCIPCA     | Candid covariance free principal component analysis.                                                   |
| CIMCA      | Covariance free incremental minor component analysis.                                                  |
| IMCA       | Incremental minor component analysis. Doesn't normalize the features so there may be stability issues. |

Citation
--------

> V. R. Kompella, M. Luciw and J. Schmidhuber. "Incremental Slow Feature Analysis: Adaptive Low-Complexity Slow Feature Updating from High-Dimensional Input Streams", Neural Computation Journal, Vol. 24 (11), pp. 2994--3024, 2012.
