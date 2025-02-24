def closest(lst, K):
    return lst[min(range(len(lst)), key = lambda i: abs(lst[i] - K))]