import torch

from ataxx import model


class Batch:
    def __init__(self, path, capacity):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.net = model.load(path, "cuda").to(memory_format=torch.channels_last)
        self.capacity = capacity
        self.graphs = {capacity: self.capture(capacity)}

    def capture(self, size):
        forward = torch.compile(self.net, fullgraph=True, dynamic=size != self.capacity)
        x = torch.zeros(size, model.PLANES, 7, 7, device="cuda").contiguous(memory_format=torch.channels_last)
        with torch.inference_mode():
            for _ in range(3):
                forward(x)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                policy, value = forward(x)
        return x, graph, policy, value

    def __call__(self, features, indices):
        n = len(indices)
        size = min(self.capacity, 1 << (n - 1).bit_length())
        if size not in self.graphs:
            self.graphs[size] = self.capture(size)
        x, graph, policy, value = self.graphs[size]
        x[:n].copy_(features[indices], non_blocking=True)
        graph.replay()
        return policy[:n].cpu().numpy(), value[:n].cpu().numpy()
