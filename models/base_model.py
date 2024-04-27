import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def extract_feat(self, *args):
        raise NotImplementedError

    def encoding(self, *args):
        raise NotImplementedError

    def decoding(self, *args):
        raise NotImplementedError


class FeatExtractor(nn.Module):
    def __init__(self):
        super(FeatExtractor, self).__init__()

    def forward(self):
        pass


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

    def forward(self):
        pass


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

    def forward(self):
        pass


class RNTrajRec(nn.Module):
    def __init__(self):
        super(RNTrajRec, self).__init__()
        self.extractor = FeatExtractor()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, *args):
        feats = self.extractor(*args)
        enc_vec = self.encoder(feats)
        output = self.decoder(enc_vec)
