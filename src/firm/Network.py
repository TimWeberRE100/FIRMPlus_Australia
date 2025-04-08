import numpy as np


def network_neighbours(network, n):
    isn_mask = np.isin(network, n)
    hasn_mask = isn_mask.sum(axis=1).astype(bool)
    joins_n = network[hasn_mask][~isn_mask[hasn_mask]]
    return joins_n


def nthary_network(network0, network_1):
    """primary, secondary, tertiary, ..., nthary"""
    """supply n-1thary to generate nthary etc."""
    networkn = -1 * np.ones((1, network_1.shape[1] + 1), dtype=np.int64)
    for row in network_1:
        _networkn = -1 * np.ones((1, network_1.shape[1] + 1), dtype=np.int64)
        joins_start = network_neighbours(network0, row[0])
        joins_end = network_neighbours(network0, row[-1])
        for n in joins_start:
            if n not in row:
                _networkn = np.vstack((_networkn, np.insert(row, 0, n)))
        for n in joins_end:
            if n not in row:
                _networkn = np.vstack((_networkn, np.append(row, n)))
        _networkn = _networkn[1:, :]
        dup = []
        # find rows which are already in network
        for i, r in enumerate(_networkn):
            for s in networkn:
                if np.setdiff1d(r, s).size == 0:
                    dup.append(i)
        # find duplicated rows within n3
        for i, r in enumerate(_networkn):
            for j, s in enumerate(_networkn):
                if i == j:
                    continue
                if np.setdiff1d(r, s).size == 0:
                    dup.append(i)
        _networkn = np.delete(_networkn, np.unique(np.array(dup, dtype=np.int64)), axis=0)
        if _networkn.size > 0:
            networkn = np.vstack((networkn, _networkn))
    networkn = networkn[1:, :]
    return networkn

def count_lines(network):
    unique, counts = np.unique(network[:, np.array([0, -1])], return_counts=True)
    if counts.size > 0:
        return counts.max()
    return 0


def generate_network(network, Nodel_int):

    # direct network connections
    network_mask = np.array([(network == j).sum(axis=1).astype(np.bool_) for j in Nodel_int]).sum(axis=0) == 2
    network = network[network_mask, :]
    networkdict = {v: k for k, v in enumerate(Nodel_int)}
    # translate into indicies rather than Nodel_int values
    basic_network = np.array([networkdict[n] for n in network.flatten()], np.int64).reshape(network.shape)
    network = basic_network.copy()
    
    trans_mask = np.zeros((len(Nodel_int), len(network)), np.bool_)
    for line, row in enumerate(network):
        trans_mask[row[0], line] = True

    directconns = -1 * np.ones((len(Nodel_int) + 1, len(Nodel_int) + 1), np.int64)
    for n, row in enumerate(network):
        directconns[*row] = n
        directconns[*row[::-1]] = n

    networks = [network]
    while True:
        n = nthary_network(network, networks[-1])
        if n.size > 0:
            networks.append(n)
        else:
            break
    triangulars = np.array([sum(range(n)) for n in range(1, len(networks) + 2)], np.int64)  # enough for now

    maxconnections = max([count_lines(network) for network in networks])

    network = -1 * np.ones((2, len(Nodel_int), triangulars[len(networks)], maxconnections), dtype=np.int64)
    for i, net in enumerate(networks):
        conns = np.zeros(len(Nodel_int), int)
        for j, row in enumerate(net):
            network[0, row[0], triangulars[i] : triangulars[i + 1], conns[row[0]]] = row[1:]
            network[0, row[-1], triangulars[i] : triangulars[i + 1], conns[row[-1]]] = row[:-1][::-1]
            conns[row[0]] += 1
            conns[row[-1]] += 1

    for i in range(network.shape[1]):
        for j in range(network.shape[2]):
            for k in range(network.shape[3]):
                if j in triangulars:
                    start = i
                else:
                    start = network[0, i, j - 1, k]
                network[1, i, j, k] = directconns[start, network[0, i, j, k]]

    directconns = directconns[:-1, :-1]
    return basic_network, network, network_mask, trans_mask, directconns, triangulars
