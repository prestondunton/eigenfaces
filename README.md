# Eigenfaces

<img src="./images/banner.PNG" alt="faces" width="100%" height="100%" align="center">

## About

This project is the term project for CSU's MATH 469: Linear Algebra II.  It is an implementation of Eigenfaces, which is a method of data reduction / facial recognition that uses eigendecomposition to store face images as a linear combination of some optimal "eigenfaces."

The full report can be read in `Eigenfaces.pdf`.  A poster summary can be found in `Poster.pdf`.  This poster has ommitted some of the content in the full report, because I couldn't fit it all.

## Introduction (Abstract)
Eigenface decomposition is the application of Principal Component Analysis (PCA) on images that contain faces in them. This decomposition allows implementers to achieve data reduction in images by calculating the vectors in high dimensional space that represent the largest variation in images. These vectors are called eigenfaces, and are the principal components of a dataset of face images. The corresponding eigenvalues represent the variance in image data each eigenface explains. Applying this to data reduction, we observe a 2.618 : 1 data reduction, but also that modern data reduction techniques are more efficient. Furthermore, we observe that by keeping the first 1589 eigenvectors in <img src="https://latex.codecogs.com/gif.latex?R^{4096}" />, we can retain 99.9% of variation in images. Finally, a discussion of this technique applied to facial recognition and machine learning is presented.

## Note
When viewing Eigenfaces.ipynb, some of the LaTex in the Calculating Eigenvectors section is missing, for an unknown reason.  If you want to see the proof as to why <img src="https://latex.codecogs.com/gif.latex?S" /> and <img src="https://latex.codecogs.com/gif.latex?A^TA" /> have the same eigenvalues, look in Eigenfaces.pdf.

## Contact

To ask me about the project, email preston.dunton@gmail.com.
