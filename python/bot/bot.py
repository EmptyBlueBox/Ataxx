import json
import os
import subprocess
import sys
import time

version = "data/Ataxx/v-@@V@@"
os.chmod(version + "/engine", 0o755)


def start(first, env=None):
    command = [version + "/engine", version + "/net.bin", "botzone", str(first), "3.85"] + [version + "/" + p for p in "@@P@@".split()]
    return subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True, bufsize=1, env=env)


def ask(engine, line):
    try:
        engine.stdin.write(line)
        engine.stdin.flush()
        return engine.stdout.readline(), engine.stdout.readline()
    except IOError:
        return "", ""


engine = start(11.6)
history = None
for line in sys.stdin:
    began = time.time()
    limit = 4.0 if history else 12.0
    if history:
        history["requests"].append(json.loads(line))
    else:
        history = json.loads(line)
    answer, marker = ask(engine, line)
    if not marker:
        engine = start(max(0.2, limit - (time.time() - began) - 1.0), dict(os.environ, ATAXX_NOVNNI="1"))
        answer, marker = ask(engine, json.dumps(history) + "\n")
    history["responses"].append(json.loads(answer)["response"])
    sys.stdout.write(answer + marker)
    sys.stdout.flush()
