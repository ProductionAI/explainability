from ampligraph.evaluation import train_test_split_no_unseen 
from ampligraph.latent_features import ScoringBasedEmbeddingModel
from ampligraph.latent_features.loss_functions import SelfAdversarialLoss
from ampligraph.utils import create_tensorboard_visualizations
import tensorflow as tf
import pandas as pd

path = '/home/jwh/workspace/explainability/data/hetionet-v1.0-edges.sif.gz'
df = pd.read_csv(path, sep='\t')

X_train, X_test = train_test_split_no_unseen(df.values, test_size=1000) 
X_train, X_valid = train_test_split_no_unseen(X_train, test_size=500)

X = {'train':X_train, 'test': X_test, 'valid': X_valid}

optim = tf.optimizers.Adam(learning_rate=0.01)
loss = SelfAdversarialLoss({'margin': 0.1, 'alpha': 5, 'reduction': 'sum'})
model = ScoringBasedEmbeddingModel(eta=5,
                                   k=200,
                                   scoring_type='TransE',
                                   seed=0)
model.compile(optimizer=optim, loss=loss)
history = model.fit(X['train'],
          batch_size=10000,
          epochs=5)

create_tensorboard_visualizations(model,
                                      entities_subset='all',
                                      loc = './embeddings_vis')
# On terminal run: tensorboard --logdir='./full_embeddings_vis' --port=8891
# Open the browser and go to the following URL: http://127.0.0.1:8891/#projector