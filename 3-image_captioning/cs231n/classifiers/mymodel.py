import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Encoder(nn.Module):
    #Encoder_CNN
    def __init__(self, embed_size):
        super(Encoder, self).__init__()
        model = models.resnet101(pretrained=True)
        resnet = models.resnet152(pretrained=True)
        self.resnet = nn.Sequential(*list(model.children())[:-1])
        #Add a fully connected layer of size (resnet.fc.in_features, embed_size)
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)
        
        
    def forward(self,image):
        with torch.no_grad():
            x=self.resnet(image)
        x=self.fc(x)
        x=self.bn(x)
        return x
        

        
# class Decoder(nn.Module):
#     #Recoder_RNN
#     def __init__(self,vocab_size, embed_size, num_hidden, num_layers, seq_length):
#         super(Decoder, self).__init__()
#         self.embed = nn.Embedding(vocab_size, embed_size)
#         self.lstm  = nn.LSTM(embed_size,num_hidden,num_layers, batch_first=True)
#         self.fc    = nn.Linear(num_hidden, vocab_size)
        
#     def forward(self,features,captions):
#         embeddings = self.embed(captions)
#         embeddings = torch.cat((features.unsqueeze(1),embeddings),1)
#         embeddings = pack_padded_sequence(embeddings,seq_length,batchfirst=True)
#         hidden, _  = self.lstm(embeddings)
#         output     = self.fc(hidden[0])
#         return output
        