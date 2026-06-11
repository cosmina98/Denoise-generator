import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler



class InterpolationSampler(object):
    def __init__(self, 
        estimator=ExtraTreesClassifier(n_estimators=300, n_jobs=-1), 
        n_output_samples_per_interpolation=1, 
        n_nearest_neighbors_per_interpolation=1, 
        min_distance_k_nearest_neighbors=1, 
        use_interpolation=True, 
        use_extrapolation=False,
        min_distance=None, 
        metric='euclidean'):
        self.estimator = estimator
        self.n_output_samples_per_interpolation = n_output_samples_per_interpolation
        self.n_nearest_neighbors_per_interpolation = n_nearest_neighbors_per_interpolation
        self.metric = metric
        self.min_distance = min_distance
        self.min_distance_k_nearest_neighbors = min_distance_k_nearest_neighbors
        self.use_interpolation = use_interpolation
        self.use_extrapolation = use_extrapolation
    
    def fit(self, X,y=None):
        self.train_X = X
        self.train_y = y
        if self.train_y is not None:
            self.estimator.fit(self.train_X, self.train_y)
        return self
    
    def interpolate(self, x_src, x_dest, n_output_samples_per_interpolation):
        samples = []
        alphas = np.linspace(0,1,n_output_samples_per_interpolation+2)[1:-1]
        if self.use_interpolation: samples += [(x_dest-x_src)*alpha+x_src for alpha in alphas]
        if self.use_extrapolation: samples += [(x_dest-x_src)*alpha+x_dest for alpha in alphas]
        samples = np.array(samples)
        return samples
        
    def sample(self, n_samples):
        if self.train_y is not None:
            samples, samples_targets = self.transform(self.train_X)
        else:
            samples = self.transform(self.train_X)

        idxs = np.random.choice(len(samples), size=n_samples)
        sel_samples = samples[idxs]
        if self.train_y is not None:
            sel_targets = samples_targets[idxs]
            return sel_samples, sel_targets
        else:
            return sel_samples
        
    def transform(self, X):
        if self.min_distance is None:
            distance_mtx = pairwise_distances(self.train_X, metric=self.metric)
            self.min_distance = np.mean(np.sort(distance_mtx, axis=1)[:,1:1+self.min_distance_k_nearest_neighbors])
        distance_mtx = pairwise_distances(X,self.train_X, metric=self.metric)
        distance_mtx[distance_mtx <= self.min_distance] = np.inf
        neighbor_idxs = np.argsort(distance_mtx, axis=1)
        samples = []
        for idx in range(X.shape[0]):
            for neighbor_idx in neighbor_idxs[idx, :self.n_nearest_neighbors_per_interpolation]:
                samples_ = self.interpolate(X[idx], self.train_X[neighbor_idx], n_output_samples_per_interpolation=self.n_output_samples_per_interpolation)
                samples.append(samples_)
        samples = np.vstack(samples)        
        if self.train_y is not None:
            samples_targets = self.estimator.predict(samples)
            return samples, samples_targets
        else:
            return samples



class EncoderDecoderInterpolationSampler(object):
    """
    A class to perform sampling using an encoder-decoder model combined with a interpolation sampler.

    Attributes:
    encoder_decoder (EncoderDecoder): An instance of an encoder-decoder model.
    interpolation_sampler (InterpolationSampler): An instance of a interpolation sampler.
    """

    def __init__(self, encoder_decoder=None, interpolation_sampler=None):
        """
        Initializes the EncoderDecoderInterpolationSampler with specified encoder-decoder and interpolation sampler.

        Parameters:
        encoder_decoder (EncoderDecoder, optional): An instance of an encoder-decoder model. 
        interpolation_sampler (InterpolationSampler): An instance of a interpolation sampler.
        """
        self.encoder_decoder = encoder_decoder
        self.interpolation_sampler = interpolation_sampler

    def fit(self, X, y=None):
        """
        Fits the encoder-decoder to the data and then fits the interpolation sampler to the encoded data.

        Parameters:
        X (array-like): The feature matrix.
        y (array-like): The target vector.

        Returns:
        self: Returns the instance itself.
        """
        # Encode the data using the encoder-decoder model
        Z = self.encoder_decoder.fit(X).encode(X)

        # Fit the interpolation sampler to the encoded data
        self.interpolation_sampler.fit(Z, y)
        if y is not None: 
            self.is_class_conditional = True
        else:
            self.is_class_conditional = False
        return self

    def sample(self, n_samples):
        """
        Generates samples using the interpolation sampler and then decodes them using the encoder-decoder model.

        Parameters:
        n_samples (int): The number of samples to generate.

        Returns:
        array-like: The generated samples in the original feature space.
        """
        # Generate samples in the encoded space using the interpolation sampler
        if self.is_class_conditional == True:
            Z, targets = self.interpolation_sampler.sample(n_samples)
        else:
            Z = self.interpolation_sampler.sample(n_samples)

        # Decode the generated samples back to the original feature space
        X = self.encoder_decoder.decode(Z)

        if self.is_class_conditional == True: return X, targets
        else: return X

    def transform(self, X):
        # Encode the data using the encoder-decoder model
        Z = self.encoder_decoder.encode(X)

        if self.is_class_conditional == True:
            Zp, targets = self.interpolation_sampler.transform(Z)
        else:
            Zp = self.interpolation_sampler.transform(Z)

        # Decode the generated samples back to the original feature space
        Xp = self.encoder_decoder.decode(Zp)

        if self.is_class_conditional == True: return Xp, targets
        else: return Xp



class KernelPCAEncoderDecoder(object):
    """
    A class to perform encoding and decoding of data using Kernel Principal Component Analysis (KernelPCA).

    Attributes:
    estimator (KernelPCA): The KernelPCA estimator used for transforming and inverse transforming the data.
    """

    def __init__(self, n_components=100):
        """
        Initializes the KernelPCAEncoderDecoder with the specified number of components.

        Parameters:
        n_components (int, optional): Number of components for KernelPCA. Default is 100.
        """
        self.scaler = StandardScaler()
        self.estimator = KernelPCA(n_components=n_components, kernel='rbf', gamma=None, fit_inverse_transform=True)

    def fit(self, data_mtx):
        """
        Fits the KernelPCA estimator to the data matrix.

        Parameters:
        data_mtx (array-like): The data matrix to fit.

        Returns:
        self: Returns the instance itself.
        """
        scaled_data_mtx = self.scaler.fit_transform(data_mtx)
        self.estimator.fit(scaled_data_mtx)
        return self

    def encode(self, data_mtx):
        """
        Transforms the data matrix into the latent space using KernelPCA.

        Parameters:
        data_mtx (array-like): The data matrix to transform.

        Returns:
        array-like: The transformed latent matrix.
        """
        scaled_data_mtx = self.scaler.fit_transform(data_mtx)
        latent_mtx = self.estimator.transform(scaled_data_mtx)
        return latent_mtx

    def decode(self, latent_mtx):
        """
        Inverse transforms the latent matrix back to the original data space using KernelPCA.

        Parameters:
        latent_mtx (array-like): The latent matrix to inverse transform.

        Returns:
        array-like: The inverse transformed data matrix.
        """
        scaled_data_mtx = self.estimator.inverse_transform(latent_mtx)
        data_mtx = self.scaler.inverse_transform(scaled_data_mtx)
        return data_mtx



class SVDEncoderDecoder(object):
    """
    A class to perform encoding and decoding of data using Truncated Singular Value Decomposition (SVD).

    Attributes:
    estimator (TruncatedSVD): The TruncatedSVD estimator used for transforming and inverse transforming the data.
    """

    def __init__(self, n_components=100):
        """
        Initializes the SVDEncoderDecoder with the specified number of components.

        Parameters:
        n_components (int, optional): Number of components for TruncatedSVD. Default is 100.
        """
        self.estimator = TruncatedSVD(n_components=n_components)

    def fit(self, data_mtx):
        """
        Fits the TruncatedSVD estimator to the data matrix.

        Parameters:
        data_mtx (array-like): The data matrix to fit.

        Returns:
        self: Returns the instance itself.
        """
        self.estimator.fit(data_mtx)
        return self

    def encode(self, data_mtx):
        """
        Transforms the data matrix into the latent space using TruncatedSVD.

        Parameters:
        data_mtx (array-like): The data matrix to transform.

        Returns:
        array-like: The transformed latent matrix.
        """
        latent_mtx = self.estimator.transform(data_mtx)
        return latent_mtx

    def decode(self, latent_mtx):
        """
        Inverse transforms the latent matrix back to the original data space using TruncatedSVD.

        Parameters:
        latent_mtx (array-like): The latent matrix to inverse transform.

        Returns:
        array-like: The inverse transformed data matrix.
        """
        data_mtx = self.estimator.inverse_transform(latent_mtx)
        return data_mtx


def ConcreteEncoderDecoderInterpolationSampler(
    n_components=10, 
    use_linear_encoder=True, 
    n_output_samples_per_interpolation=1,
    n_nearest_neighbors_per_interpolation=1, 
    min_distance=None, 
    min_distance_k_nearest_neighbors=1):
    # Choose the encoder-decoder model based on the use_linear_encoder flag
    if use_linear_encoder:
        encoder_decoder = SVDEncoderDecoder(n_components=n_components)
    else:
        encoder_decoder = KernelPCAEncoderDecoder(n_components=n_components)

    interpolation_sampler = InterpolationSampler(
        estimator=ExtraTreesClassifier(n_estimators=300, n_jobs=-1), 
        n_output_samples_per_interpolation=n_output_samples_per_interpolation, 
        n_nearest_neighbors_per_interpolation=n_nearest_neighbors_per_interpolation,  
        metric='euclidean', 
        min_distance=min_distance, 
        min_distance_k_nearest_neighbors=min_distance_k_nearest_neighbors)
    
    # Create an instance of EncoderDecoderInterpolationSampler with the encoder-decoder and interpolation_sampler
    sampler = EncoderDecoderInterpolationSampler(encoder_decoder, interpolation_sampler)

    return sampler