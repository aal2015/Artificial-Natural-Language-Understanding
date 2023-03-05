import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.data.utils import get_tokenizer

import stanza

import random, math

device = torch.device('cpu')

################################## Loading Vocab ##################################
vocab_transform = torch.load('vocab_transform.pth')

###################################### Model ######################################
# Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, dropout):
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, bidirectional = True)
        self.fc = nn.Linear(hid_dim * 2, hid_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_len):
        embedded = self.dropout(self.embedding(src))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len.to('cpu'), enforce_sorted=False)              
        packed_outputs, hidden = self.rnn(packed_embedded)        
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs) 
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
        return outputs, hidden

# Attention
class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        
        self.v = nn.Linear(hid_dim, 1, bias = False)
        self.W = nn.Linear(hid_dim,     hid_dim)
        self.U = nn.Linear(hid_dim * 2, hid_dim)
                
    def forward(self, hidden, encoder_outputs, mask):
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)        
        energy = torch.tanh(self.W(hidden) + self.U(encoder_outputs))       
        attention = self.v(energy).squeeze(2)
        attention = attention.masked_fill(mask, -1e10) 
        return F.softmax(attention, dim = 1)

# Decoder
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout, attention):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.gru = nn.GRU((hid_dim * 2) + emb_dim, hid_dim)
        self.fc = nn.Linear((hid_dim * 2) + hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs, mask):
        input = input.unsqueeze(0) 
        embedded = self.dropout(self.embedding(input)) 
        a = self.attention(hidden, encoder_outputs, mask)      
        a = a.unsqueeze(1) 
        encoder_outputs = encoder_outputs.permute(1, 0, 2)        
        weighted = torch.bmm(a, encoder_outputs)       
        weighted = weighted.permute(1, 0, 2)  
        rnn_input = torch.cat((embedded, weighted), dim = 2)
        output, hidden = self.gru(rnn_input, hidden.unsqueeze(0))

        assert (output == hidden).all()
        
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        
        prediction = self.fc(torch.cat((output, weighted, embedded), dim = 1))

        return prediction, hidden.squeeze(0), a.squeeze(1)

# Putting them together (become Seq2Seq!)
class Seq2SeqPackedAttention(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.device = device
        
    def create_mask(self, src):
        mask = (src == self.src_pad_idx).permute(1, 0)  #permute so it's the same shape as attention
        return mask
        
    def forward(self, src, src_len, max_trg_len, teacher_forcing_ratio = 0.5):
        
        #src = [src len, batch size]
        #src_len = [batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time
                    
        batch_size     = src.shape[1]
        trg_len        = max_trg_len
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        #tensor to store attentiont outputs from decoder
        attentions = torch.zeros(trg_len, batch_size, src.shape[0]).to(self.device)
        
        #encoder_outputs is all hidden states of the input sequence, back and forwards
        #hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src, src_len)
                
        #first input to the decoder is the <sos> tokens
        input_ = torch.tensor([2])
        
        mask = self.create_mask(src)
        #mask = [batch size, src len]
                
        for t in range(1, trg_len):
            
            #insert input token embedding, previous hidden state, all encoder hidden states 
            #  and mask
            #receive output tensor (predictions) and new hidden state
            output, hidden, attention = self.decoder(input_, hidden, encoder_outputs, mask)
            #output    = [batch size, output dim]
            #hidden    = [batch size, hid dim]
            #attention = [batch size, src len]
            
            #place predictions in a tensor holding predictions for each token
            outputs[t] = output
            
            #place attentions in a tensor holding attention for each token
            attentions[t] = attention
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            #get the highest predicted token from our predictions
            top1 = output.argmax(1) 
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input_ = trg[t] if teacher_force else top1
            
        return outputs, attentions

SRC_LANGUAGE = 'hi'
TRG_LANGUAGE = 'en'

def initialize_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)

PAD_IDX, SOS_IDX, EOS_IDX = 1, 2, 3

input_dim   = len(vocab_transform[SRC_LANGUAGE])
output_dim  = len(vocab_transform[TRG_LANGUAGE])
emb_dim     = 256  
hid_dim     = 512  
dropout     = 0.5
SRC_PAD_IDX = PAD_IDX

attn = Attention(hid_dim)
enc  = Encoder(input_dim,  emb_dim,  hid_dim, dropout)
dec  = Decoder(output_dim, emb_dim,  hid_dim, dropout, attn)

model = Seq2SeqPackedAttention(enc, dec, SRC_PAD_IDX, device).to(device)
model.apply(initialize_weights)

path = './models/Seq2SeqPackedAttention.pt' 
model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
model.eval()

#################################### Inference ####################################
token_transform = {}

# stanza.download('hi')
hindi_tokenizer = stanza.Pipeline('hi', processors='tokenize', download_method=None)

def tokenizeHindiSent(text):
    doc = hindi_tokenizer(text)
    
    for sentence in doc.sentences:
        hindi_tokens = [token.text for token in sentence.tokens]
    return hindi_tokens

token_transform[TRG_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_md')
token_transform[SRC_LANGUAGE] = tokenizeHindiSent

def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids):
    return torch.cat((torch.tensor([SOS_IDX]), 
                      torch.tensor(token_ids), 
                      torch.tensor([EOS_IDX])))

# src and trg language text transforms to convert raw strings into tensors indices
text_transform = {}
for ln in [SRC_LANGUAGE, TRG_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln], #Tokenization
                                               vocab_transform[ln], #Numericalization
                                               tensor_transform) # Add BOS/EOS and create tensor

def translate(hindi_text):
    src_text = text_transform[SRC_LANGUAGE](hindi_text).to(device)
    src_text = src_text.reshape(-1, 1)  #because batch_size is 1
    text_length = torch.tensor([src_text.size(0)]).to(dtype=torch.int64)
    
    max_trg_text_len = int(src_text.shape[0] * 1.5)

    with torch.no_grad():
        output, attentions = model(src_text, text_length, max_trg_text_len, 0) #turn off teacher forcing

    output = output.squeeze(1)
    output = output[1:]

    output_max = output.argmax(1)

    mapping = vocab_transform[TRG_LANGUAGE].get_itos()

    english_tokens = []
    for token in output_max:
        en_token = mapping[token.item()]
        
        if en_token == "<eos>":
            break

        english_tokens.append(en_token)
    
    translated_en_sent = " ".join(english_tokens)
    return translated_en_sent 