import timm
import torch
from pathlib import Path
from typing import Optional
import torch.nn as nn

class LinBnDrop(nn.Module):
    """
    A linear layer that optionally incorporates batch norm and dropout on input
    and a ReLU activation. Inspired by fast.ai.

    Takes (*, H_in) of dim (*, n_in) and produces (*, H_out) of dim (*, n_out)
    """
    def __init__(self, n_in, n_out, dropout: float=0., bn: bool=False, act: bool=True):
        """
        n_in: int
            dim of input (actually, the size of the last dimension)
        n_out: int
            dim of output (# of neurons in the layer)
        dropout: float=0.
            rate of dropout applied to the input (disabled by default)
        bn: bool=False
            whether to use batch norm on input (disabled by default)
        act: bool=True
            whether to use ReLU activation (enabled by default)
        """
        super().__init__()
        
        self.bn = nn.BatchNorm1d(n_in) if bn else None
        self.drop = nn.Dropout(dropout) if dropout > 0 else None
        self.fc = nn.Linear(n_in, n_out)
        self.act = nn.ReLU() if act else None

    def forward(self, x):
        if self.bn:
            x = self.bn(x)

        if self.drop:
            x = self.drop(x)

        x = self.fc(x)

        if self.act:
            x = self.act(x)

        return x
    
class SimpleModel(nn.Module):
    """
    A simple model that takes only numeric inputs
    predicts a numerical value for them.

    Inputs: (*, H_in) of dim (*, n_in)
    Outputs: (*, H_out) of dim (*, 1)
    """
        
    def __init__(self, 
                 n_numeric: int, 
                 hidden_layers: list[int], 
                 dropout_in: float=0., 
                 dropout: float=0.,
                 batch_norm_in: bool=False,
                 batch_norm: bool=False,
                 ):
        """
        n_numeric: int
            dim of input (actually, the size of the last dimension)
        hidden_layers: list[int]
            list containing the numbers of neurons in the hidden layers
        dropout_in: float
            rate of dropout on inputs (before the first dropout layer)
        dropout: float
            rate of dropout for all further layers
        batch_norm_in: bool
            whether to apply batch norm to inputs
        batch_norm: bool
            whether to apply batch norm in the further layers
        """
        super().__init__()
        
        self.lin_layers = []

        n_out = hidden_layers[0]
        self.lin_layers.append(LinBnDrop(n_numeric, n_out, dropout_in, batch_norm_in))
        
        n_in = n_out
        for n_out in hidden_layers[1:]:
            self.lin_layers.append(LinBnDrop(n_in, n_out, dropout, batch_norm))
            n_in = n_out

        self.lin_layers = nn.ModuleList(self.lin_layers)
        
        self.head = LinBnDrop(n_in, 1, dropout, batch_norm, act=False)


    def forward(self, numeric, spatial):
        x = numeric
        for layer in self.lin_layers:
            x = layer(x)
        
        out = self.head(x)

        return out
    

"""
Here we create a model that uses resnet18 to process spatial data
"""
def create_timm_body(arch:str, pretrained: bool=True, cut: Optional[int]=None, in_chans:int=3, drop_rate:float=0.):
    """
    This function uses the timm library to create a vision model.

    arch:
        the model architecture (from timm) to use
    pretrained:
        whether to use a pretrained model
    cut:
        how many layers to use (precisely: how many model.children() elements to use)
    in_chans:
        how many input channels to use (warning: you can use numbers other than 1 or 3, but then don't expect excellent behavior without further training! - it just gives a better initialization)
    drop_rate: 
        the dropout rate to use for training this model

    Returns a timm model.
    """
    model = timm.create_model(arch, pretrained=pretrained, num_classes=0, global_pool='', in_chans=in_chans, drop_rate=drop_rate)
    if isinstance(cut, int): 
        return nn.Sequential(*list(model.children())[:cut])
    elif callable(cut): 
        return cut(model)
    else: 
        raise ValueError("cut must be either integer or function")

# How many channels resnet18 blocks give on output
resnet18_channels = {
    1: 64,
    2: 64,
    3: 128,
    4: 256,
    5: 512,
}

# Where to take cuts to get a specific number of resnet blocks from timm
resnet18_cuts = {
    1: 3,
    2: 5,
    3: 6,
    4: 7,
    5: 8
}


class NumericWithResnet18(nn.Module):
    """
    This is the main model developed for the POC.
    Its structure is described in the summary presentation. Shortly:
    
    SPATIAL -> RESNET18 -> AVGPOOL -> FC (multiple) -> HEAD
    NUMERICAL -> FC (num) ------------------^
    i.e., the penultimate FC layers are fed the outputs of AVGPOOL on RESNET on spatial data
    and the output of the single FC layer on NUMERICAL data.
    """
        
    def __init__(self, 
                 n_numeric: int, 
                 numeric_layer_size: int,
                 hidden_layers_sizes: list[int], 
                 spatial_feature_channels: int=0,
                 spatial_input_size: int=128,
                 dropout_in: float=0., 
                 dropout_resnet: float=0.,
                 dropout_resnet_features: float=0.,
                 dropout: float=0.,
                 batch_norm_in: bool=False,
                 batch_norm: bool=False,
                 resnet_layers: int=2,
                 freeze_resnet: bool=False,
                 zero_resnet: bool=False,
                 separate_channels: bool=False,
                 pretrained_resnet: bool=True,
                 ):
        """
        n_numeric:
            dimension of the numeric inputs
        numeric_layer_size:
            dimension of the single hidden layer that processes numeric inputs
        hidden_layers_sizes:
            list containing the numbers of neurons in the hidden layers in the penultimate part of the network
        spatial_feature_channels: 
            how many channels are in the spatial inputs
            setting this to 0 is equivalent to setting zero_resnet=True, produces a numeric-only model
        spatial_input_size:
            size of the spatial inputs [i.e., their dimensions are (*, spatial_feature_channels, spatial_input_size, spatial_input_size)]
        dropout_in:
            rate of dropout for numeric inputs - FC (num)
        dropout_resnet:
            rate of dropout for RESNET18
        dropout_resnet_features:
            rate of dropout applied to the output of AVGPOOL
        dropout:
            rate of dropout for the penultimate FC layers
        batch_norm_in:
            whether to use batch norm for numeric inputs
        batch_norm:
            whether to use batch norm in the penultimate FC layers
        resnet_layers:
            how many convolutional blocks of resnet to use (out of 5)
        freeze_resnet:
            if set to True, resnet block will not be trained
        zero_resent:
            this makes the resnet inactive, simulates it not being present at all
            (effectively, this makes the model a numeric-only model)
        separate_channels:
            whether to process each channel separately through resnet
            (did some experiments, but it doesn't seem like a good idea 
            - unless you're using pretrained weights and frozen resnet)
        pretrained_resnet:
            whether to initialize the resnet with pretrained weights (or random otherwise)
        """
        super().__init__()
        
        """
        Processing variables
        """
        # The model has an option to not use spatial data at all
        if zero_resnet:
            spatial_feature_channels = 0
        if spatial_feature_channels == 0:
            zero_resnet = True

        # Save the variables for later
        self.zero_resnet = zero_resnet
        self.separate_channels = separate_channels
        self.spatial_channels = spatial_feature_channels
        self.n_numeric = n_numeric

        # If using separate channels, resnet has 1 input channel
        if self.separate_channels:
            in_channels = 1
        else:
            in_channels = self.spatial_channels

        """
        CNN part
        """
        if zero_resnet:
            # If not using spatial data at all, do not initialize the resnet block
            self.resnet = None
            self.pool = None
            self.drop_resnet_features = None
            self.flatten = None
            self.res_n_out = 0
        else:
            self.resnet = create_timm_body(
                'resnet18.a1_in1k', 
                pretrained=pretrained_resnet,
                drop_rate=dropout_resnet,
                cut=resnet18_cuts[resnet_layers],
                in_chans=in_channels
                )

            if freeze_resnet:
                self.resnet.requires_grad_(False)
            
            divide_by = 2**resnet_layers
            if spatial_input_size % divide_by != 0:
                raise ValueError(f"Spatial input size (got spatial_input_size={spatial_input_size}) has to be divisible by 2^resnet_layers (got resnet_layers={resnet_layers})!")
            self.pool = nn.AvgPool2d(spatial_input_size // 2**resnet_layers)

            self.drop_resnet_features = nn.Dropout(dropout_resnet_features)
            self.flatten = nn.Flatten()

            self.res_n_out = resnet18_channels[resnet_layers]

        """
        The single FC layer for numeric inputs
        """
        n_out = numeric_layer_size
        self.fc_num = LinBnDrop(self.n_numeric, n_out, dropout_in, batch_norm_in)
        
        """
        Fully connected part
        """
        self.lin_layers = []

        if self.separate_channels:
            n_in = n_out + self.res_n_out * self.spatial_channels
        else:
            n_in = n_out + self.res_n_out

        for n_out in hidden_layers_sizes:
            self.lin_layers.append(LinBnDrop(n_in, n_out, dropout, batch_norm))
            n_in = n_out

        self.lin_layers = nn.ModuleList(self.lin_layers)
        
        self.head = LinBnDrop(n_in, 1, dropout, batch_norm, act=False)


    def forward(self, numeric, spatial):
        # Single FC layer for numeric inputs
        # assert numeric.size(1) == self.n_numeric
        num = self.fc_num(numeric)

        # CNN part - parallel resnets
        if self.zero_resnet:
            x  = num
        else:
            spatial_features = []

            # Gather spatial features
            if self.separate_channels:
                for i in range(self.spatial_channels):
                    spatial_feature = spatial[:,i].unsqueeze(1)
                    spatial_features.append(spatial_feature)
            else:
                spatial_features = [spatial]

            # Process all of the features through resnet
            resnet_outputs = []
            for spatial_feature in spatial_features:
                res = self.resnet(spatial_feature)
                res_pool = self.pool(res)
                res_pool_flat = self.flatten(self.drop_resnet_features(res_pool))

                resnet_outputs.append(res_pool_flat)
            x = torch.cat([num] + resnet_outputs, dim=1)
            
        # A list of FC layers
        for layer in self.lin_layers:
            x = layer(x)
        
        out = self.head(x)

        return out


class SpatialOnlyResnet18(nn.Module):

    def __init__(self,
                 n_numeric: int,
                 hidden_layers_sizes: list[int],
                 spatial_feature_channels: int = 0,
                 spatial_input_size: int = 128,
                 dropout_resnet: float = 0.,
                 dropout_resnet_features: float = 0.,
                 dropout: float = 0.,
                 batch_norm: bool = False,
                 resnet_layers: int = 2,
                 freeze_resnet: bool = False,
                 pretrained_resnet: bool = True,
                 ):
        """
        n_numeric:
            dimension of the numeric inputs
        numeric_layer_size:
            dimension of the single hidden layer that processes numeric inputs
        hidden_layers_sizes:
            list containing the numbers of neurons in the hidden layers in the penultimate part of the network
        spatial_feature_channels:
            how many channels are in the spatial inputs
            setting this to 0 is equivalent to setting zero_resnet=True, produces a numeric-only model
        spatial_input_size:
            size of the spatial inputs [i.e., their dimensions are (*, spatial_feature_channels, spatial_input_size, spatial_input_size)]
        dropout_in:
            rate of dropout for numeric inputs - FC (num)
        dropout_resnet:
            rate of dropout for RESNET18
        dropout_resnet_features:
            rate of dropout applied to the output of AVGPOOL
        dropout:
            rate of dropout for the penultimate FC layers
        batch_norm_in:
            whether to use batch norm for numeric inputs
        batch_norm:
            whether to use batch norm in the penultimate FC layers
        resnet_layers:
            how many convolutional blocks of resnet to use (out of 5)
        freeze_resnet:
            if set to True, resnet block will not be trained
        zero_resent:
            this makes the resnet inactive, simulates it not being present at all
            (effectively, this makes the model a numeric-only model)
        separate_channels:
            whether to process each channel separately through resnet
            (did some experiments, but it doesn't seem like a good idea
            - unless you're using pretrained weights and frozen resnet)
        pretrained_resnet:
            whether to initialize the resnet with pretrained weights (or random otherwise)
        """
        super().__init__()

        """
        Processing variables
        """
        self.n_numeric = 0
        # Save the variables for later
        self.spatial_channels = spatial_feature_channels

        in_channels = self.spatial_channels

        """
        CNN part
        """

        self.resnet = create_timm_body(
            'resnet18.a1_in1k',
            pretrained=pretrained_resnet,
            drop_rate=dropout_resnet,
            cut=resnet18_cuts[resnet_layers],
            in_chans=in_channels
        )

        if freeze_resnet:
            self.resnet.requires_grad_(False)

        divide_by = 2 ** resnet_layers
        if spatial_input_size % divide_by != 0:
            raise ValueError(
                f"Spatial input size (got spatial_input_size={spatial_input_size}) has to be divisible by 2^resnet_layers (got resnet_layers={resnet_layers})!")
        self.pool = nn.AvgPool2d(spatial_input_size // 2 ** resnet_layers)

        self.drop_resnet_features = nn.Dropout(dropout_resnet_features)
        self.flatten = nn.Flatten()

        self.res_n_out = resnet18_channels[resnet_layers]


        """
        Fully connected part
        """
        self.lin_layers = []

        n_in = self.res_n_out

        for n_out in hidden_layers_sizes:
            self.lin_layers.append(LinBnDrop(n_in, n_out, dropout, batch_norm))
            n_in = n_out

        self.lin_layers = nn.ModuleList(self.lin_layers)

        self.head = LinBnDrop(n_in, 1, dropout, batch_norm, act=False)

    def forward(self, numerical, spatial):

        # Gather spatial features
        spatial_features = [spatial]

        # Process all of the features through resnet
        resnet_outputs = []
        for spatial_feature in spatial_features:
            res = self.resnet(spatial_feature)
            res_pool = self.pool(res)
            res_pool_flat = self.flatten(self.drop_resnet_features(res_pool))

            resnet_outputs.append(res_pool_flat)
        #x = torch.cat([num] + resnet_outputs, dim=1)
        x = torch.cat(resnet_outputs, dim=1)

        # A list of FC layers
        for layer in self.lin_layers:
            x = layer(x)

        out = self.head(x)

        return out

class DummyModel(nn.Module):
    """
    A dummy model for simple tests on numerical data
    """
    def __init__(self, n_in: int, n_2: int):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_2)
        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(n_2, 1)

    def forward(self, numeric, spatial):
        out1 = self.fc1(numeric)
        act1 = self.relu(out1)
        
        out2 = self.fc2(act1)
        return out2
    

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(3)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

class CustomConvNet(nn.Module):
    def __init__(self, num_blocks, n_features, hidden_layers_sizes, dropout, batch_norm):
        super().__init__()
        self.blocks = nn.ModuleList()
        image_size = 128
        in_channels = n_features
        for i in range(num_blocks):
            if i == 0:
                out_channels = 64
            else:
                out_channels = in_channels * 2
            self.blocks.append(ConvBlock(in_channels, out_channels))
            in_channels = out_channels

        conv_output_size = image_size // (3 ** num_blocks)

        n_in = out_channels * conv_output_size * conv_output_size

        self.lin_layers = []

        for n_out in hidden_layers_sizes:
            self.lin_layers.append(LinBnDrop(n_in, n_out, dropout, batch_norm))
            n_in = n_out

        self.lin_layers = nn.ModuleList(self.lin_layers)

        self.head = LinBnDrop(n_in, 1, act=False)

    def forward(self, numerical_inputs, spatial_inputs):
        x = spatial_inputs
        for block in self.blocks:
            x = block(x)

        x = x.view(x.size(0), -1)

        for layer in self.lin_layers:
            x = layer(x)

        out = self.head(x)

        return out

def save_torch_state_dict(
        model: nn.Module,
        save_filepath: Path,
):
    """
    This saves the torch state dict to {save_filepath}.pt
    """
    save_filepath = save_filepath.parent / (save_filepath.name + '.pt')
    save_filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_filepath)


def load_torch_state_dict(
        model: nn.Module,
        save_filepath: Path,
):
    """
    This loads the torch state dict from {save_filepath}.pt
    """
    save_filepath = save_filepath.parent / (save_filepath.name + '.pt')
    model.load_state_dict(torch.load(save_filepath))
    return model

