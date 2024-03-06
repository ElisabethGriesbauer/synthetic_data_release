# ## tuning no. epochs and batch_size for real data I d20
# ## (per epoch we have n/batch_size steps)
# from pandas import DataFrame
# import os
# os.chdir('./synthetic_data_release-master')

# from utils.logging import LOGGER

# from generative_models.generative_model import GenerativeModel
# from sdv.single_table import CTGANSynthesizer
# from sdv.metadata import SingleTableMetadata
# # from ctgan import CTGAN
# import plotly.express as px
# import pandas as pd

# from utils.datagen import load_local_data_as_df

# rawPop, metadata = load_local_data_as_df('./data/real_data_I_d20')

# metadata = SingleTableMetadata()
# metadata.detect_from_dataframe(data=rawPop)


# ctgan = CTGANSynthesizer( metadata,
#    enforce_rounding=False,
#    epochs=500,
#    batch_size = 100,
#    verbose=True)

# ctgan.fit(data=rawPop)
# loss_values = ctgan._model.loss_values

# loss_values_reformatted = pd.melt(
#    loss_values,
#    id_vars=['Epoch'],
#    var_name='Loss Type'
# )

# fig = px.line(loss_values_reformatted, x="Epoch", y="value", color="Loss Type", title='Epoch vs. Loss')
# fig.show()

#-----------------------------

from pandas import DataFrame, concat
import random

from utils.logging import LOGGER

from generative_models.generative_model import GenerativeModel
from sdv.single_table.ctgan import TVAESynthesizer
from sdv.metadata import SingleTableMetadata


class TVAE(GenerativeModel):
    """A wrapper for a tabular variational autoencoder"""
    def __init__(self, metadata,
                #  enforce_min_max_values=True,
                #  enforce_rounding=False,
                #  locales=['en_US'],
                 epochs=1500, # tuned by hand for real data I d20
                 batch_size=400, # tuned by hand for real data I d20
                 embedding_dim=2, # tuned by hand for real data I d20
                 compress_dims=(128, 128),
                 decompress_dims=(128, 128),
                 l2scale=1e-5,
                 loss_factor=2,
                 verbose = True,
                 multiprocess=False,
                 cuda=True):

        self.metadata = metadata
        # self.enforce_min_max_values = enforce_min_max_values
        # self.enforce_rounding = enforce_rounding
        # self.locales = locales
        self.cuda = cuda
        self.epochs = epochs
        self.batch_size = batch_size
        self.compress_dims = compress_dims
        self.decompress_dims = decompress_dims
        self.embedding_dim = embedding_dim
        self.l2scale = l2scale
        self.loss_factor = loss_factor
        self.verbose = verbose

        self.datatype = DataFrame

        self.multiprocess = bool(multiprocess)

        self.infer_ranges = True
        self.trained = False

        self.__name__ = 'TVAE'

    def fit(self, data, *args):
        """Train a tabular variational autoencoder.
        Input data is assumed to be of shape (n_samples, n_features)
        See https://github.com/DAI-Lab/SDGym for details"""
        assert isinstance(data, self.datatype), f'{self.__class__.__name__} expects {self.datatype} as input data but got {type(data)}'

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=data)

        self.synthesiser = TVAESynthesizer(metadata=metadata,
                #  enforce_min_max_values = self.enforce_min_max_values,
                #  enforce_rounding=self.enforce_rounding,
                #  locales = self.locales,
                 cuda = self.cuda,
                 epochs=self.epochs,
                 batch_size=self.batch_size,
                 compress_dims = self.compress_dims,
                 decompress_dims = self.decompress_dims,
                 embedding_dim = self.embedding_dim,
                 l2scale = self.l2scale,
                 loss_factor = self.loss_factor)
                #  verbose = True)
                #  pac = self.pac)

        self.synthesiser._model_kwargs = self.synthesiser._model_kwargs | {'verbose': self.verbose}

        ## does not work because TVAE is initialized only in fit function of TVAESynthesiser
        # self.synthesiser._model.set_random_state(123)
        
        ## not needed for tvae it seems
        # if len(args) > 0:
        #     # Merge the additional data frames using pandas.concat or another appropriate method
        #     data = concat([data] + list(args), axis=0, ignore_index=True)

        LOGGER.debug(f'Start fitting {self.__class__.__name__} to data of shape {data.shape}...')
        self.synthesiser.fit(data)

        LOGGER.debug(f'Finished fitting')
        self.trained = True

        return self

    def generate_samples(self, nsamples):
        """Generate random samples from the fitted Gaussian distribution"""
        assert self.trained, "Model must first be fitted to some data."

        LOGGER.debug(f'Generate synthetic dataset of size {nsamples}')
        randint = random.randint(1, 100000000000000000000)

        # use first alternative for MIA:
        # synthetic_data = self.synthesiser.sample(num_rows=nsamples, output_file_path=f'./tmp_samples/temp{randint}')
        synthetic_data = self.synthesiser.sample(num_rows=nsamples, output_file_path=None)

        return synthetic_data
    
    def set_params(self, **params):
        for param, value in params.items():
            if hasattr(self, param):
                setattr(self, param, value)
            else:
                raise ValueError(f"Invalid parameter: {param}")
            
    
    def transform(self, X):
        # You might need to adjust this logic based on your specific generative model
        # return self.synthesiser.sample(num_rows=len(X), output_file_path=None)
        randint = random.randint(1, 100000000000000000000)

        ## use first alternative for MIA:
        return self.synthesiser.sample(num_rows=len(X), output_file_path=f'./tmp_samples/temp{randint}')
