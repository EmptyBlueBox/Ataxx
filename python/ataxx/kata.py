import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnx import numpy_helper


def rms(x, w):
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(x.dtype) * w


def gpool(x, value):
    m = x.mean((2, 3))
    return torch.cat([m, m * -0.7, m * (0.39) if value else x.amax((2, 3))], 1)


class Block(nn.Module):
    def __init__(self, c, heads, ffn):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.Parameter(torch.ones(c)), nn.Parameter(torch.ones(c))
        self.qkv = nn.Linear(c, 3 * c, bias=False)
        self.out = nn.Linear(c, c, bias=False)
        self.ffn1 = nn.Linear(c, 2 * ffn, bias=False)
        self.ffn2 = nn.Linear(ffn, c, bias=False)

    def forward(self, x, cos, sin):
        n, s, c = x.shape
        q, k, v = self.qkv(rms(x, self.norm1)).view(n, s, 3, self.heads, -1).unbind(2)
        rot = lambda t: torch.stack([-t[..., 1::2], t[..., 0::2]], -1).flatten(-2)
        q, k = q * cos + rot(q) * sin, k * cos + rot(k) * sin
        a = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        x = x + self.out(a.transpose(1, 2).reshape(n, s, c))
        h, g = self.ffn1(rms(x, self.norm2)).chunk(2, -1)
        return x + self.ffn2(F.silu(h) * g)


class KataNet(nn.Module):
    def __init__(self, blocks=12, c=384, heads=6, ffn=1024):
        super().__init__()
        self.spatial = nn.Conv2d(22, c, 3, padding=1, bias=False)
        self.glob = nn.Linear(19, c, bias=False)
        self.blocks = nn.ModuleList([Block(c, heads, ffn) for _ in range(blocks)])
        self.register_buffer("cos", torch.zeros(49, 1, c // heads))
        self.register_buffer("sin", torch.zeros(49, 1, c // heads))
        self.final = nn.Parameter(torch.zeros(4, c))
        self.p1, self.g1 = nn.Conv2d(c, 48, 1, bias=False), nn.Conv2d(c, 48, 1)
        self.pg, self.ppass = nn.Linear(144, 48, bias=False), nn.Linear(144, 4, bias=False)
        self.p2b, self.p2 = nn.Parameter(torch.zeros(48)), nn.Conv2d(48, 4, 1, bias=False)
        self.v1 = nn.Conv2d(c, 96, 1)
        self.v2 = nn.Linear(288, 128)
        self.value, self.misc, self.more = nn.Linear(128, 3), nn.Linear(128, 10), nn.Linear(128, 8)
        self.own = nn.Conv2d(96, 1, 1, bias=False)

    def forward(self, spatial, glob):
        n = len(spatial)
        x = (self.spatial(spatial) + self.glob(glob)[:, :, None, None]).flatten(2).transpose(1, 2)
        for b in self.blocks:
            x = b(x, self.cos, self.sin)
        mean, std, gamma, beta = self.final
        t = F.silu((x - mean) / std * gamma + beta).transpose(1, 2).reshape(n, -1, 7, 7)
        g = gpool(F.silu(self.g1(t)), False)
        p = F.silu(self.p1(t) + self.pg(g)[:, :, None, None] + self.p2b[:, None, None])
        policy = torch.cat([self.p2(p).flatten(2), self.ppass(g)[:, :, None]], 2)
        v = F.silu(self.v1(t))
        h = F.silu(self.v2(gpool(v, True)))
        return policy, self.value(h), self.misc(h), self.more(h), self.own(v)


def load(path):
    m = onnx.load(path)
    w = {t.name: torch.from_numpy(numpy_helper.to_array(t).copy()) for t in m.graph.initializer}
    consts = {n.output[0]: torch.from_numpy(numpy_helper.to_array(n.attribute[0].t).copy()) for n in m.graph.node if n.op_type == "Constant" and n.attribute[0].name == "value"}
    weight = {n.name: next(w[i] for i in n.input if i in w) for n in m.graph.node if n.op_type in ("MatMul", "Gemm") and any(i in w for i in n.input)}
    net = KataNet()
    sd = net.state_dict()
    sd["spatial.weight"] = w["model.conv_spatial.weight"]
    sd["glob.weight"] = weight["/model/linear_global/MatMul"].T
    for i in range(12):
        b = f"/model/blocks.{i}/"
        sd[f"blocks.{i}.norm1"], sd[f"blocks.{i}.norm2"] = w[f"model.blocks.{i}.norm1.weight"], w[f"model.blocks.{i}.norm2.weight"]
        sd[f"blocks.{i}.qkv.weight"] = torch.cat([weight[b + f"{k}_proj/MatMul"].T for k in "qkv"])
        sd[f"blocks.{i}.out.weight"] = weight[b + "out_proj/MatMul"].T
        sd[f"blocks.{i}.ffn1.weight"] = torch.cat([weight[b + "ffn_linear1/MatMul"].T, weight[b + "ffn_linear_gate/MatMul"].T])
        sd[f"blocks.{i}.ffn2.weight"] = weight[b + "ffn_linear2/MatMul"].T
    tables = [consts[n.input[0]] for n in m.graph.node if n.name in ("/model/blocks.0/Reshape_4", "/model/blocks.0/Reshape_5")]
    sd["cos"], sd["sin"] = tables[0].view(49, 1, -1), tables[1].view(49, 1, -1)
    final = {n.name: next(w[i] for i in n.input if i in w) for n in m.graph.node if n.name.startswith("/model/norm_trunkfinal/")}
    sd["final"] = torch.stack([final[f"/model/norm_trunkfinal/{k}"].flatten() for k in ("Sub", "Div", "Mul", "Add")])
    sd["p1.weight"], sd["g1.weight"] = w["model.policy_head.conv1p.weight"], w["model.policy_head.conv1g.weight"]
    sd["g1.bias"] = w["model.policy_head.biasg.beta"].flatten()
    sd["pg.weight"], sd["ppass.weight"] = weight["/model/policy_head/linear_g/MatMul"].T, weight["/model/policy_head/linear_pass/MatMul"].T
    sd["p2b"], sd["p2.weight"] = w["model.policy_head.bias2.beta"].flatten(), w["model.policy_head.conv2p.weight"]
    sd["v1.weight"], sd["v1.bias"] = w["model.value_head.conv1.weight"], w["model.value_head.bias1.beta"].flatten()
    for a, b in (("v2", "linear2"), ("value", "linear_valuehead"), ("misc", "linear_miscvaluehead"), ("more", "linear_moremiscvaluehead")):
        sd[f"{a}.weight"], sd[f"{a}.bias"] = w[f"model.value_head.{b}.weight"], w[f"model.value_head.{b}.bias"]
    sd["own.weight"] = w["model.value_head.conv_ownership.weight"]
    net.load_state_dict(sd)
    return net.eval()
