import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

PLANES, KINDS, SQ = 14, 17, 49


def pooled(x):
    return torch.cat([x.mean((2, 3)), x.amax((2, 3))], 1)


class Block(nn.Module):
    def __init__(self, c, pool):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)
        self.pool = nn.Linear(2 * c, c) if pool else None

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        if self.pool is not None:
            y = y + self.pool(pooled(y))[:, :, None, None]
        return F.relu(x + self.bn2(self.conv2(y)))


class Net(nn.Module):
    def __init__(self, blocks=6, channels=96):
        super().__init__()
        c = channels
        self.config = dict(blocks=blocks, channels=channels)
        self.stem = nn.Sequential(nn.Conv2d(PLANES, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU())
        self.blocks = nn.Sequential(*[Block(c, i % 3 == 1) for i in range(blocks)])
        self.policy1 = nn.Conv2d(c, 32, 1)
        self.policy_pool = nn.Linear(2 * c, 32)
        self.policy2 = nn.Conv2d(32, KINDS, 1)
        self.value1 = nn.Conv2d(c, 32, 1)
        self.value2 = nn.Linear(64, 64)
        self.value3 = nn.Linear(64, 4)
        self.ownership = nn.Conv2d(c, 1, 1)

    def trunk(self, x):
        return self.blocks(self.stem(x))

    def heads(self, t):
        p = F.relu(self.policy1(t) + self.policy_pool(pooled(t))[:, :, None, None])
        policy = self.policy2(p).flatten(1)
        value = self.value3(F.relu(self.value2(pooled(F.relu(self.value1(t))))))
        return policy, value

    def forward(self, x):
        return self.heads(self.trunk(x))

    def train_forward(self, x):
        t = self.trunk(x)
        return (*self.heads(t), self.ownership(t).flatten(1))


def save(net, path):
    torch.save(dict(config=net.config, state=net.state_dict()), path)


def load(path, device="cpu"):
    data = torch.load(path, map_location=device)
    net = Net(**data["config"]).to(device)
    net.load_state_dict(data["state"])
    return net.eval()


def export(net, path):
    net = net.cpu().float().eval()
    torch.onnx.export(net, torch.zeros(1, PLANES, 7, 7), path, input_names=["x"], output_names=["policy", "value"],
                      dynamic_axes={"x": {0: "n"}, "policy": {0: "n"}, "value": {0: "n"}}, opset_version=17, dynamo=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt", required=True)
    parser.add_argument("--onnx", required=True)
    args = parser.parse_args()
    export(load(args.pt), args.onnx)


if __name__ == "__main__":
    main()
