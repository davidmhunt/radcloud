from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
from torchvision.transforms import CenterCrop
from torch.nn import functional as F
from torch.nn import BatchNorm2d
import torch

class _Block(Module):


    def __init__(self, in_channels, out_channels, batch_norm = True):
        """Initialize a standard convolutional block with 2 convolutional layers

        Args:
            in_channels (int): the number of channels on the input
            out_channels (int): the number of channels at the output
            batch_norm(bool): on True enabled batch normalization
        """
        super().__init__()

        #store the convolution and ReLU layers
        self.conv1 = Conv2d(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=(3,3),
                            stride=1,
                            padding=1)
        self.relu = ReLU()
        self.conv2 = Conv2d(in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=(3,3),
                            stride=1,
                            padding=1)
        
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.batch_norm1 = BatchNorm2d(num_features=out_channels)
            self.batch_norm2 = BatchNorm2d(num_features=out_channels)
    
    def forward(self,x):
        """perform a forward pass of the Block convolution

        Args:
            x (Tensor): #TODO: get the dimmensions of the input tensor

        Returns:
            Tensor: output tensor after convolutions #TODO: get dimmensions of output tensor
        """

        x = self.conv1(x)
        if self.batch_norm:
            x = self.batch_norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        if self.batch_norm:
            x = self.batch_norm2(x)
        x = self.relu(x)

        return x

#defining the encoder and decoder blocks
class _EncoderBlock(Module):

    def __init__(self,in_channels,out_channels):
        """Creat new Encoder Block class

        Args:
            in_channels (int): the number of input channels for the encoder block
            out_channels (int): the number of output channels for the encoder block
        """
        super().__init__()

        self.conv_block = _Block(in_channels,out_channels)
        self.pool = MaxPool2d(2)
    
    def forward(self,x):
        """Perform a forward pass on the Encoder Block

        Args:
            x (Tensor): input tensor

        Returns:
            (Tensor,Tensor): (x, skip features) where skip_features is the output before max_pooling and x is the output after max_pooling
        """

        skip_features = self.conv_block(x)
        x = self.pool(skip_features)

        #TODO: check to make sure that you can output two variables. If not, output them as a list
        return (x,skip_features)

class _DecoderBlock(Module):

    def __init__(self, in_channels, out_channels):
        """Create new Decoder block module

        Args:
            in_channels (int): the number of input channels
            out_channels (int): the number of output channels
        """
        super().__init__()

        self.conv_block = _Block(in_channels,out_channels)
        self.up_conv = ConvTranspose2d(in_channels= in_channels,
                                       out_channels= out_channels,
                                       kernel_size=2,
                                       stride=2)
    
    def forward(self,x,skip_features):
        """Perform a forward pass of the decoder block

        Args:
            x (Tensor): the input tensor into the decoder (decoder performs upconvolution)
            skip_features (Tensor): a tensor of skip features from the encoder with the same input height and width

        Returns:
            Tensor: the output tensor from the decoder block
        """
        
        #up convolve the input
        x = self.up_conv(x)

        #crop and then concatenate the skip_features from the encoder
        #skip_features = self.crop(skip_features,x)
        x = torch.cat([x,skip_features],dim=1)

        #perform the convolutions
        x = self.conv_block(x)

        return x
    
    def crop(self, skip_features, x):
        """Crops the skip_features tensor if it does not have the same height and width as x

        Args:
            skip_features (Tensor): Tensor of skip features
            x (Tensor): input tensor used as reference to crop the skip features tensor

        Returns:
            Tensor : A center-cropped version of the skip features tensor with the same height and width as x
        """
        
        #TODO: determine how this works
        (_,_,H,W) = x.shape
        skip_features = CenterCrop([H,W])(skip_features)

        return skip_features

#defining the encoder and decoder modules
class Encoder(Module):

    def __init__(self,input_channels = 3, output_channels=(16,32,64)):
        """Initialize a new Encoder module which is a chain of EncoderBlocks

        Args:
            input_channels (int,optional): the number of input channels into the encoder. Defaults to 3
            output_channels (tuple, optional): tuple of the number of output channels for each encoder in the encoder block. Defaults to (3,16,32,64).
        """
        super().__init__()

        self.channels = [input_channels]
        self.channels.extend(output_channels)
        self.encoder_blocks = ModuleList(
            [_EncoderBlock(
                in_channels=self.channels[i],
                out_channels=self.channels[i+1])
            for i in range(len(self.channels) - 1)])
    
    def forward(self,x):
        """Perform a forward pass of the Encoder

        Args:
            x (Tensor): Input tensor into the encoder

        Returns:
            tuple: (x,skip_feature_outputs) where x is the output tensor of the final encoder block and skip_feature_outputs is a list of skip features from each encoder in the order [1st encoder, 2nd encoder, Nth encoder]
        """
        skip_feature_outputs = []

        for encoder in self.encoder_blocks:

            #get the output (max pooled) and skip layers (before max pooling) from the encoder
            (x,skip_features) = encoder(x)
            
            #append the skip features to the list
            skip_feature_outputs.append(skip_features)
        
        return (x,skip_feature_outputs)
    
class Decoder(Module):

    def __init__(self,input_channels=(64,32),output_channels=16):
        """Initialize a new decoder module

        Args:
            input_channels (tuple, optional): list of the number of input channels into each decoder block. Defaults to (64,32).
            output_channels (int,optional): the number of output channels at the end of the decoder block
        """
              
        super().__init__()

        #get a list of channels
        self.channels = []
        self.channels.extend(input_channels)
        self.channels.append(output_channels)

        self.decoder_blocks = ModuleList(
            [_DecoderBlock(
                in_channels=self.channels[i],
                out_channels=self.channels[i + 1]
            ) for i in range(len(self.channels) - 1)]
        )

    def forward(self,x,skip_feature_outputs):
        """Perform a forward pass of the Decoder module

        Args:
            x (Tensor): Input tensor into the decoder block
            skip_feature_outputs (Tensor): a list of skip feature tensors which will be given to the decoders in the following order [1st tensor -> 1st decoder, ... Nth tensor -> Nth decoder]

        Returns:
            Tensor: output tensor from the decoder module
        """

        for i in range(len(self.channels) - 1):

            x = self.decoder_blocks[i](x,skip_feature_outputs[i])

        return x

class unet(Module):

    def __init__(self,
                 encoder_input_channels = 3,
                 encoder_out_channels = (16,32,64),
                 decoder_input_channels=(128,64,32),
                 decoder_out_channels = 16,
                 output_channels = 1,
                 retain_dimmension=True,
                 input_dimmensions=(128,128)):
        """Initialize a new unet architecture

        Args:
            encoder_input_channels (int, optional): number of channels at the input to the encoder. Defaults to 3.
            encoder_out_channels (tuple, optional): number of channels at the output of each encoder block in the encoder. Defaults to (16,32,64).
            decoder_input_channels (tuple, optional): number of channels at the input of each decoder block in the decoder. Defaults to (128,64,32).
            decoder_out_channels (int, optional): number of channels at the output of the decoder. Defaults to 16.
            output_channels (int, optional): number of output channels to the model. Defaults to 1.
            retain_dimmension (bool, optional): on True, the output retains the same dimmensions as the input. Defaults to True.
            input_dimmensions (tuple, optional): The dimmensions of each input to the model. Defaults to (128,64).
        """
        super().__init__()

        #initialize the encoder and decoder
        self.encoder = Encoder(
            input_channels=encoder_input_channels,
            output_channels=encoder_out_channels
        )

        self.decoder = Decoder(
            input_channels=decoder_input_channels,
            output_channels = decoder_out_channels
        )

        self.retain_dimmension = retain_dimmension
        self.out_channels = output_channels
        self.output_size = input_dimmensions

        #initialize layer between encoder and decoder
        self.encoder_decoder_link = _Block(encoder_out_channels[-1],decoder_input_channels[0]) 
        
        #initialize the regression head
        self.head = Conv2d(
            in_channels= decoder_out_channels,
            out_channels= output_channels,
            kernel_size= 1
        )

    def forward(self,x):

        #pass input through encoder
        (x,skip_feature_outputs) = self.encoder(x)

        #pass through link layer
        x = self.encoder_decoder_link(x)
        
        #pass through the decoder
        x = self.decoder(x,skip_feature_outputs[::-1])

        #get the output map using the head
        x = self.head(x)

        #resize output to match if requested
        if self.retain_dimmension:
            x = F.interpolate(x,self.output_size)

        return x



