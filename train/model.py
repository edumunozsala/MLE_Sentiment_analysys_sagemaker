import torch.nn as nn
import torch

class LSTMClassifier(nn.Module):
    """
    This is the simple RNN model we will be using to perform Sentiment Analysis.
    """

    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        """
        Initialize the model by settingg up the various layers.
        """
        super(LSTMClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        ### UPDATED ###
        # Added a n_layers and dropout parameters
        # We should probably include these parameters as parameters but for simplicity
        # we devide to "hardcode" the values, instead of not being a best practice
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, bidirectional=True)
        self.dense = nn.Linear(in_features=hidden_dim*2, out_features=1)
        # Added a dropout layer
        self.dropout = nn.Dropout(0.3)
        
        self.sig = nn.Sigmoid()
        
        self.word_dict = None

    def forward(self, x):
        """
        Perform a forward pass of our model on some input.
        """
        x = x.t()
        lengths = x[0,:]
        reviews = x[1:,:]
        ### UPDATED ###
        embeds = self.dropout(self.embedding(reviews))
        #lstm_out, _ = self.lstm(embeds)
        #out = self.dense(lstm_out)
        ## UPDATED ##
        lstm_out, (hidden, cell) = self.lstm(embeds)
        #concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        #and apply dropout
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        out = self.dense(hidden)
        
        ### UPDATED ###
        # Dropout
        #dout = self.dropout(out)
        #out = dout[lengths - 1, range(len(lengths))]
        return self.sig(out.squeeze())