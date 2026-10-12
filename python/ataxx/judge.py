import numpy as np

from ataxx.selfplay import library

RESULTS = {0: "black", 1: "white", -1: "draw"}


def status(moves):
    flat = np.array(moves, np.int32).reshape(-1)
    verdict = np.zeros(4, np.int32)
    library().adjudicate(flat.ctypes.data, len(moves), verdict.ctypes.data)
    finish, winner, error, _ = verdict
    if not finish:
        return "ongoing"
    if error:
        return ("illegal", len(moves))
    return RESULTS[winner]
