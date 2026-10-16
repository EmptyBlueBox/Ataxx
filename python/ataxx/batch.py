import torch

from ataxx import model


class Batch:
    def __init__(self, path, capacity):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.net = model.load(path, "cuda").to(memory_format=torch.channels_last)
        forward = torch.compile(self.net, fullgraph=True, dynamic=False)
        self.x = torch.zeros(capacity, model.PLANES, 7, 7, device="cuda").contiguous(memory_format=torch.channels_last)
        with torch.inference_mode():
            for _ in range(3):
                forward(self.x)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.policy, self.value = forward(self.x)

    def __call__(self, features, indices):
        n = len(indices)
        self.x[:n].copy_(features[indices], non_blocking=True)
        self.graph.replay()
        return self.policy[:n].cpu().numpy(), self.value[:n].cpu().numpy()
