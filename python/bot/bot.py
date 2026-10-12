import os

runtime = "data/Ataxx/rt-@@RT@@"
version = "data/Ataxx/v-@@V@@"
loader = runtime + "/ld-linux-x86-64.so.2"
os.chmod(loader, 0o755)
os.chmod(version + "/engine", 0o755)
os.execv(loader, [loader, "--library-path", runtime, version + "/engine", version + "/net.onnx", "botzone", "11.1", "3.7", version + "/book.txt"])
