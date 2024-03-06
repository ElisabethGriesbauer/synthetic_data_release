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
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata


class CTGAN(GenerativeModel):
    """A conditional generative adversarial network for tabular data"""
    def __init__(self, metadata,
                 epochs=1000, # tuned by hand for real data I d20
                 batch_size=150, # tuned by hand for real data I d20
                 enforce_rounding=False,
                 verbose=True,
                 dis_dim=(256, 256),
                 discriminator_decay=1e-6,
                 discriminator_lr=2e-4,
                 discriminator_steps=1,
                 embedding_dim=128, 
                 generator_decay=1e-6,
                 # l2scale=1e-6,
                 gen_dim=(256, 256),
                 generator_lr=2e-4,
                 log_frequency = True,
                 pac = 10,
                 multiprocess=False):

        self.metadata = metadata
        self.enforce_rounding = enforce_rounding
        self.epochs = epochs
        self.verbose = verbose
        self.batch_size = batch_size
        self.dis_dim = dis_dim
        self.discriminator_decay = discriminator_decay
        self.discriminator_lr = discriminator_lr
        self.discriminator_steps = discriminator_steps
        self.embedding_dim = embedding_dim
        self.generator_decay = generator_decay
        self.gen_dim = gen_dim
        self.generator_lr = generator_lr
        self.log_frequency = log_frequency
        self.pac = pac

        self.datatype = DataFrame

        self.multiprocess = bool(multiprocess)

        self.infer_ranges = True
        self.trained = False

        self.__name__ = 'CTGAN'

    def fit(self, data, *args):
        """Train a generative adversarial network on tabular data.
        Input data is assumed to be of shape (n_samples, n_features)
        See https://github.com/DAI-Lab/SDGym for details"""
        assert isinstance(data, self.datatype), f'{self.__class__.__name__} expects {self.datatype} as input data but got {type(data)}'

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=data)

        self.synthesiser = CTGANSynthesizer(metadata=metadata,
                 epochs=self.epochs,
                 batch_size=self.batch_size,
                 verbose=self.verbose,
                 enforce_rounding=self.enforce_rounding,
                 discriminator_dim=self.dis_dim,
                 discriminator_decay=self.discriminator_decay,
                 discriminator_lr=self.discriminator_lr,
                 discriminator_steps=self.discriminator_steps,
                 embedding_dim=self.embedding_dim, 
                 generator_decay=self.generator_decay,
                 generator_dim=self.gen_dim,
                 generator_lr=self.generator_lr,
                 log_frequency = self.log_frequency,
                 pac = self.pac)
        
        if len(args) > 0:
            # Merge the additional data frames using pandas.concat or another appropriate method
            data = concat([data] + list(args), axis=0, ignore_index=True)

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

        ## use first alternative for MIA:
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
        return self.synthesiser.sample(num_rows=len(X), output_file_path=None)
