import torch
import torch.nn as nn

class ScaledEuclidDistanceAttention(nn.Module):
    def __init__(self, dim_hidden: int, num_heads: int, qkv_bias: bool=False):
        super().__init__()

        assert dim_hidden % num_heads == 0

        self.num_heads = num_heads

        dim_head = dim_hidden // num_heads

        # Scale Value of Softmax
        self.scale = dim_head ** -0.5

        self.proj_q = nn.Linear( dim_hidden, dim_hidden, bias=qkv_bias)
        self.proj_k = nn.Linear( dim_hidden, dim_hidden, bias=qkv_bias)
        self.proj_v = nn.Linear( dim_hidden, dim_hidden, bias=qkv_bias)

        self.proj_out = nn.Linear(dim_hidden, dim_hidden)

    def forward(self, q, k, v, mask = None ):

        q = self.proj_q( q )
        k = self.proj_k( k )
        v = self.proj_v( v )

        q = q.view( q.size(0), q.size(1), self.num_heads, -1 ).permute( 0,2,1,3 )
        k = k.view( k.size(0), k.size(1), self.num_heads, -1 ).permute( 0,2,1,3 )
        v = v.view( v.size(0), v.size(1), self.num_heads, -1 ).permute( 0,2,1,3 )

        qk_dis = torch.cdist( q, k, p = 2 ) * self.scale # cdist が pytorch の距離関数です。
        attn =  1 / ( qk_dis + 1e-9 )
        # Confirmation of learning was carried out mask = None
        if mask is not None:
            attn = attn + torch.unsqueeze( torch.unsqueeze( mask, dim = 1 ), dim = 3 ).to(torch.float16) * -1e9
        attn = ( attn ).softmax(dim=-1)  

        x = attn.matmul(v)

        x = x.permute(0, 2, 1, 3).flatten(2)
        x = self.proj_out(x)

        return x
        

if __name__ == "__main__":

    num_batch = 8
    q_seq = 300
    k_seq = 100
    dim_hidden = 512
    num_heads = 8

    func2 = ScaledEuclidDistanceAttention( dim_hidden, num_heads )

    q = torch.randn( ( num_batch, q_seq, dim_hidden))
    k = torch.randn( ( num_batch, k_seq, dim_hidden))
    v = k
    mask = torch.randint(low=0, high=2, size=(num_batch,q_seq)).to( torch.bool )

    x = func2( q,k,v,mask)

    print( x.size() )
