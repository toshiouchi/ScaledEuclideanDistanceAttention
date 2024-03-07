import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, dim_hidden: int, n_head: int, dropout: float=0.1, qkv_bias: bool=False):
        super().__init__()
        self.n_head = n_head

        self.proj_q = nn.Linear( dim_hidden, dim_hidden, bias=qkv_bias)
        self.proj_k = nn.Linear( dim_hidden, dim_hidden, bias=qkv_bias)
        self.proj_v = nn.Linear( dim_hidden, dim_hidden, bias=qkv_bias)
        self.proj_out = nn.Linear(dim_hidden, dim_hidden)

        self.qkv_attention = ScaledEuclidDistanceAttention( dim_hidden, n_head, dropout )
        
    def forward( self,query,key,value, attn_mask = None ):

        q = self.proj_q( query )
        k = self.proj_k( key )
        v = self.proj_v( value )
        
        output = self.qkv_attention(q, k, v, attn_mask)

        output = self.proj_out( output )
        
        return output 

    
class ScaledEuclidDistanceAttention(nn.Module):
    def __init__(self, dim_hidden: int, num_heads: int, dropout: float=0.1):
        super().__init__()

        assert dim_hidden % num_heads == 0

        self.num_heads = num_heads

        dim_head = dim_hidden // num_heads

        # Scale Value of Softmax
        self.scale = dim_head ** -0.5
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None ):

        q = q.view( q.size(0), q.size(1), self.num_heads, -1 ).permute( 0,2,1,3 )
        k = k.view( k.size(0), k.size(1), self.num_heads, -1 ).permute( 0,2,1,3 )
        v = v.view( v.size(0), v.size(1), self.num_heads, -1 ).permute( 0,2,1,3 )

        qk_dis = torch.cdist( q, k, p = 2 ) * self.scale # cdist is distance function of pytorch
        attn =  1 / ( qk_dis + 1e-9 )
        # Check of learning was carried out with mask = None
        if mask is not None:
            attn = attn + torch.unsqueeze( torch.unsqueeze( mask, dim = 0 ), dim = 0 ).to(torch.float16) * -1e9
        attn = ( attn ).softmax(dim=-1)  

        attn = self.dropout( attn )
        
        x = attn.matmul(v)

        x = x.permute(0, 2, 1, 3).flatten(2)

        return x

if __name__ == "__main__":

    num_batch = 8
    q_seq = 300
    k_seq = 100
    dim_hidden = 512
    num_heads = 8

    func2 = MultiHeadAttention( dim_hidden, num_heads )

    q = torch.randn( ( num_batch, q_seq, dim_hidden))
    k = torch.randn( ( num_batch, k_seq, dim_hidden))
    v = k
    mask = torch.randint(low=0, high=2, size=(q_seq,k_seq)).to( torch.bool )

    x = func2( q,k,v,mask)

    print( x.size() )

    mask = torch.randint(low=0, high=2, size=(num_batch,q_seq)).to( torch.bool )

    x = func2( q,k,v,mask)

    print( x.size() )
