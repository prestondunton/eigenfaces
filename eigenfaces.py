import math
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator

def plot_images(arr, titles = None, n_col = 6):

    # all images are flattened, so we must unpack them
    
    if arr.ndim == 1:
        # a single greyscale image
        arr = arr.reshape((int(math.sqrt(arr.shape[0])), int(math.sqrt(arr.shape[0]))))
        plt.imshow(arr, cmap="gray")
        
    elif arr.ndim == 2:
        if arr.shape[1] == 3:
            # a single color image
            arr = arr.reshape((int(math.sqrt(arr.shape[0])), int(math.sqrt(arr.shape[0])), 3))
            plt.imshow(arr)
        else:
            # multiple greyscale images
            arr = arr.reshape((arr.shape[0], int(math.sqrt(arr.shape[1])), int(math.sqrt(arr.shape[1]))))
            plot_portraits(arr, titles, color=False, n_col = n_col)
            
    elif arr.ndim == 3:
        # multiple color images
        arr = arr.reshape((arr.shape[0], int(math.sqrt(arr.shape[1])), int(math.sqrt(arr.shape[1])), 3))
        plot_portraits(arr, titles, color=True, n_col = n_col)
        
    else:
        raise RuntimeError("I have no idea what you want me to do.")
        
def plot_portraits(images, titles = None, color = False, n_col = 6):
    """
    Code adapted from:
    Acar, Nev. “Eigenfaces: Recovering Humans from Ghosts.” Medium, Towards Data Science, 2 Sept. 2018, 
    towardsdatascience.com/eigenfaces-recovering-humans-from-ghosts-17606c328184.
    """
    
    n_row = math.ceil(images.shape[0] / n_col)
    
    plt.figure(figsize=(2.2 * n_col, 2.2 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.20)
    
    for i in range(images.shape[0]):
        plt.subplot(n_row, n_col, i + 1)
        
        if color:
            plt.imshow(images[i])
        else:
            plt.imshow(images[i], cmap="gray")
        
        if titles is not None:
            plt.title(titles[i])
        
        plt.xticks(())
        plt.yticks(())

def load_data(dir, num_images, shuffle=False, random_state = 42):
    image_file_names = os.listdir(dir)[0:num_images]
    image_paths = [dir + "/" + file_name for file_name in image_file_names]

    images = np.array([plt.imread(path) for path in image_paths])
    image_labels = [name[:name.find('0')-1].replace("_", " ") for name in image_file_names]

    # flatten the images into vectors
    if images.ndim == 4:
        # color images
        new_shape = (images.shape[0], images.shape[1]*images.shape[2], 3)
    else:
        # greyscale images
        new_shape = (images.shape[0], images.shape[1]*images.shape[2])
    
    images = np.reshape(images, new_shape)
    image_labels = np.array(image_labels)
    
    if shuffle:
        np.random.seed(random_state)
        indecies = np.random.permutation(len(images))
        return images[indecies], image_labels[indecies]
    else:
        return images, image_labels

class Eigenfaces(BaseEstimator,TransformerMixin):
    def __init__(self, num_eigenvectors = None, prop_var_retained = None):
        
        self.mean = None
        self.eigenvalues = None
        self.eigenvectors = None
        
        self.var_props = None
        self.cumu_var_props = None
        
        self.basis = None
        self.num_eigenvectors = num_eigenvectors
        self.prop_var_retained = prop_var_retained
        
        self.fit_images = None
        
    def fit(self, X, y = None):
        """
        Eigen decomposes the faces images specified by the numpy array 
        parameter X (long form: rows are observations, columns are variables)
        Computes min(M,N^2) eigenvectors / eigenvalues where
        X has M observations and N^2 dimensions
        
        The mean face vector is stored in self.mean
        The eigenfaces are stored in self.eigenvectors
        The eigenvalues are stored in self.eigenvalues
        
        The proportion of variance each eigenvector explains is stored in self.var_props
        The cumulative propoprtions of variance are stored in self.cumu_var_props
        
        The basis is set using a call to Eigenvaces._set_basis() which uses class parameters, or
        all of the eigenvectors if no parameters were given.
        """
        
        # mean center the data
        mean = np.mean(X, axis=0)
        X_bar = X - mean
        
        # put into wide form for calculations
        X_bar = X_bar.T
        
        n_squared = X_bar.shape[0] # number of dimensions of each image
        m = X_bar.shape[1] # number of images
        self.fit_images = m
        
        if n_squared < m:
            S = np.cov(X_bar) # an (n_squared x n_squred) matrix, same as np.cov(X)
            values, vectors = np.linalg.eig(S)
            
        else:
            A = math.sqrt(1/(m-1)) * X_bar
            ATA = np.matmul(A.T, A) # an (m x m) matrix
            
            values, vectors = np.linalg.eig(ATA)
            
            vectors = np.matmul(A, vectors)
            
        # put the eigenvectors into long form
        vectors = vectors.T
        
        # sort by eignevalue descending
        values, vectors = zip(*sorted(zip(values, vectors), reverse=True))
        values = np.array(values)
        vectors = np.array(vectors)
        
        # normalize vectors to be unit length
        norms = np.linalg.norm(vectors, axis=1)
        norms = norms.reshape(norms.shape[0],1)
        vectors = vectors / norms
        
        self.mean = mean
        self.eigenvalues = values
        self.eigenvectors = vectors
        
        self.var_props = self.eigenvalues / np.sum(self.eigenvalues)
        self.cumu_var_props = np.cumsum(self.var_props)
        
        # an orthonormal basis
        # basis is set using constructor parameters, or all eigenvectors if no parameters given
        if self.num_eigenvectors is None and self.prop_var_retained is None:
            self._set_basis(prop_var_retained = 1.0)
        else:
            self._set_basis(self.num_eigenvectors, self.prop_var_retained)
        
        return
    
    def _set_basis(self, num_eigenvectors = None, prop_var_retained = None):
        """
        Sets the basis for the face space.  This can be done in one of two methods, 
        listed below in order of priority.  self.basis is an array of indecies into self.eigenvectors.
        
        If num_eigenvectors is specified, the basis is just the first num_eigenvectors eigenvectors
        stored in self.eigenvectors.
        
        If prop_variance is specified, the basis will be made up of enough eigenvectors to meet the
        specified proportion of total variance of the original images.
        """
        
        if self.eigenvectors is None:
            raise RuntimeError("Eigenfaces haven't been calculated yet! Run Eigenfaces.dempose() to calculate eigenfaces.")
        
        if num_eigenvectors is not None:
            if num_eigenvectors <= 0 or num_eigenvectors > len(self.eigenvectors):
                raise ValueError("At least 1 and no more than " + str(len(self.eigenvectors)) + " eigenvectors " \
                "can be chosen from self.eigenvectors.")
            
            self.basis = np.arange(0,num_eigenvectors)
            
        elif prop_var_retained is not None:
            if prop_var_retained < 0 or prop_var_retained > 1:
                raise ValueError("Specified proportion of variance is not in the range [0,1].")
            elif prop_var_retained == 1:
                self.basis = np.arange(0,len(self.eigenvalues))
            else:
                index = np.where(self.cumu_var_props > prop_var_retained)[0][0]
                self.basis = np.arange(0, index+1)
            
        else:
            raise ValueError("Please specify at least one method for choosing a basis.")        
            
    def transform(self, X, y = None):
        """
        Calculates the scalars used in linear combinations of the self.basis
        vectors to create a projection of the column vetors of X onto self.basis
        
        X is in long form with observations as rows and dimensions as columns
        """
        
        if self.basis is None:
            raise RuntimeError("Basis hasn't been calculated yet!  Run Eigenfaces.fit() to calculate eigenvectors")
        
        return np.matmul(X, self.eigenvectors[self.basis].T)
    
    def reconstruct(self, projection_scalars):
        """
        Reconstructs an array of images using self.basis and an
        array of the scalars used in a linear combination of the 
        basis vectors.
        
        The numpy array projection is in long form with observations
        as rows and linear combination scalars accross columns
        """
        
        if self.basis is None:
            raise RuntimeError("Basis hasn't been calculated yet!  Run Eigenfaces.fit() to calculate eigenvectors,'\
                                ' and optionally run Eigenfaces.set_basis() to subset those basis vectors.")
        
        return np.matmul(projection_scalars, self.eigenvectors[self.basis])
    
    def project(self, X):
        """
        Projects the images specified by the numpy array parameter X
        (long form: rows are observations, columns are variables)
        onto the face space, the space spanned by the eigenvectors 
        stored in self.basis.  This is the transforming part.  The projected
        vectors are returned.
        """
        
        return self.reconstruct(self.transform(X))
    
    def fit_transform(self, X, y = None):
        """
        Calculates eigenvectors / eigenvalues by calling self.fit(X)
        This is the fitting part.
        
        Projects the images specified by the numpy array parameter X
        (long form: rows are observations, columns are variables)
        onto the face space, the space spanned by the eigenvectors 
        stored in self.basis.  This is the transforming part.  The projected
        vectors are returned.
        """
        
        self.fit(X, y)
        
        return self.transform(X, y)
    
    def save_reduced_dataset(self, X, replace = True):
    
        root_folder = './reduced_datasets'
        if not os.path.exists(root_folder):
            os.makedirs(root_folder)

        path = str(len(self.basis)) + "_" + str(self.fit_images) 
        if not os.path.exists(root_folder + '/' + path):
            os.makedirs(root_folder + '/' + path)

        if replace or not os.path.exists(root_folder + '/' + path + '/' + 'eigenfaces.csv'):
            eigenfaces = pd.DataFrame(self.eigenvectors[self.basis])
            eigenfaces.to_csv(root_folder + '/' + path + "/" + "eigenfaces.csv", header = False, index = False)

        if replace or not os.path.exists(root_folder + '/' + path + '/' + 'projections.csv'):
            projections = pd.DataFrame(self.transform(X))
            projections.to_csv(root_folder + '/' + path + "/" + "projections.csv", header = False, index = False)