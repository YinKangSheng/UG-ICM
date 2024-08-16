import torch
import torch.nn as nn


class BetaMlp(nn.Module):
    def __init__(self, channels=512, act_layer=nn.ReLU):
        super(BetaMlp, self).__init__()
        self.output_features = channels
        self.fc1 = nn.Linear(21, self.output_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(self.output_features, self.output_features)

    def forward(self, x):

        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        return x

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        # 将 self.embed_fns 中的每个函数应用于 inputs，并将结果堆叠起来
        stacked_outputs = torch.stack([fn(inputs) for fn in self.embed_fns])

        # 对堆叠后的张量进行转置操作
        transposed_outputs = stacked_outputs.permute(1, 0)
        return transposed_outputs

def get_embedder(multires, i=0):
    if i == -1:
        return lambda x: x, 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': 1,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)

    def embed(x, eo=embedder_obj): return eo.embed(x)

    return embed, embedder_obj.out_dim

# def build_global_conditioning():
#     beta = torch.randn(1)  # Dummy input, replace with actual input tensor
#
#     embed_fn, input_ch = get_embedder(multires=10, i=0)
#     beta_mlp = BetaMlp()
#     fourier_features = embed_fn(beta)
#     fourier_features_mlp = beta_mlp(fourier_features)
#
#     print(fourier_features_mlp)
#
#     return beta_mlp  # Return the model instance

class GlobalConditioning(nn.Module):
    def __init__(self):
        super(GlobalConditioning, self).__init__()
        self.beta_mlp = BetaMlp()
        self.embed_fn, self.input_ch = get_embedder(multires=10, i=0)  # Assuming get_embedder is defined elsewhere

    def forward(self, beta):
        fourier_features = self.embed_fn(beta)
        fourier_features_mlp = self.beta_mlp(fourier_features)
        return fourier_features_mlp