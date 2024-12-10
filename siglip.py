import torch 
import torch.nn as nn 


class SiglipVisionConfig:
    def __init__(self,hidden_size = 768):
        super().__init__()

class SiglipMLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config 
        self.fc1 = nn.Linear(config.hidden_size,config.intermidiate_size)
        self.fc2 = nn.Linear(config.intermidiate_size,config.hidden_size)
    def forward(self,hidden_states:torch.Tensor)->torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states,approximate="tanh")
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class SiglipEncoderLayer(nn.Module):
    def __init__(self,config:SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim,eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim,eps=config.layer_norm_eps)

    def forward(self,hidden_states:torch.Tensor)->torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states,_ = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual+hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual +  hidden_states
        return hidden_states 

class SiglipAttention(nn.module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim,self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim,self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim,self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim,self.embed_dim)
    def forward(self,hidden_states:torch.Tensor)->Tuple[torch.Tensor,Optional[torch.Tensor]]:
        batch_size,seq_len,_ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        key_states = query_states.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        value_states = query_states.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)

        attn_weights = (torch.matmul(query_states,key_states.transpose(2,3))*self.scale)

        if attn_weights.size() != (batch_size,self.num_heads,seq_len,seq_len):
            raise ValueError(
                f"Attention weights should be of size:{(batch_size,self.num_heads,seq_len,seq_len)} but is:{attn_weights.size()}"
            )
        attn_weights = nn.functional.softmax(attn_weights,dim=1,dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights,p=self.dropout,training=self.training)
        attn_output = torch.matmul(attn_weights,value_states)

        if attn_output.size() != (batch-size,self.num_heads,seq_len,self.head_dim):
            raise  ValueError(
                 f"Attention weights should be of size:{(batch_size,self.num_heads,seq_len,seq_len)} but is:{attn_weights.size()}"
            )
        attn_output = attn_output.transpose(1,2).contiguous()
        #reshape the dimensions into one single token
        attn_output = attn_output.reshape(batch_size,seq_len,self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output,attn_weights
        
class SiglipVisionEmbeddings(nn.Module):
    def __init__(self,config:SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size,
        self.image_size = config.image_size,
        self.patch_size = config.image_size


        self.patch_embedding = nn.Conv2d(
            in_channels = config.num_channels,
            out_channels = self.embed_dim,
            kernel_size = self.patch_size,
            stride = self.patch_size,
            padding = "valid"
        )

        self.num_patches = (self.image_size//self.patch_size)**2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions,self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1.-1)),
            persistent = False,
        )

    def forward(self,pixel_values:torch.FloatTensor)->torch.Tensor:
        _,_,height,width = pixel_values.shape
        # get image patch embeddings through a cnn 
        patch_embeds = self.patch_embedding(pixel_values)
        # flatten the tensor
 
        
        embeddings = patch_embeds.flatten(2)
         # from [batch_size,embed_dim,num_patches] -> [batch_size,num_patches,emb
        embeddings = embeddings.transpose(1,2)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings

class SiglipEncoder(nn.Module):
    def __init__(self,config:SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SiglipEncoder(config) for x in range(config.num_hidden_layers)]
        )
    def forward(self,inputs_embeds:torch.Tensor)-> torch.Tensor:
        hidden_states = inputs_embeds

        for x in self.layers:
            hidden_states = encoder_layer(hidden_states)
        return hidden_states
        
class SiglipVisionTransformer(nn.Module):
    def __init__(self,config:SiglipVisionConfig):
        super(). __init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim,eps=config.layer_norm_eps)

    def forward(self,pixel_values:torch.Tensor)-> torch.Tensor:
        hidden_states = self.embeddings(pixel_values)
        last_hidden_state = self.encoder(input_embeds=hidden_states)
        last_hidden_state = self.post_layernorm(last_hidden_state)
 
        return last_hidden_state


class SiglipVisionModel(nn.Module):
    def __init__(self,config:SiglipVisionConfig):

        super(). __init__() 
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self,pixel_values)->tuple:
        return self.vision_model(pixel_values = pixel_values)

