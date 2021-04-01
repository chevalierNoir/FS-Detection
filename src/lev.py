def iterative_levenshtein(s, t, costs=(1, 1, 1)):
    """ 
    Computes Levenshtein distance between the strings s and t.
    For all i and j, dist[i,j] will contain the Levenshtein 
    distance between the first i characters of s and the 
    first j characters of t

    s: source, t: target
    costs: a tuple or a list with three integers (d, i, s)
           where d defines the costs for a deletion
                 i defines the costs for an insertion and
                 s defines the costs for a substitution
    return: 
    H, S, D, I: correct chars, number of substitutions, number of deletions, number of insertions
    """
    rows = len(s)+1
    cols = len(t)+1
    deletes, inserts, substitutes = costs
    
    dist = [[0 for x in range(cols)] for x in range(rows)]
    H, D, S, I = 0, 0, 0, 0
    for row in range(1, rows):
        dist[row][0] = row * deletes
    for col in range(1, cols):
        dist[0][col] = col * inserts
        
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row-1] == t[col-1]:
                cost = 0
            else:
                cost = substitutes
            dist[row][col] = min(dist[row-1][col] + deletes,
                                 dist[row][col-1] + inserts,
                                 dist[row-1][col-1] + cost)
    row, col = rows-1, cols-1
    while row != 0 or col != 0:
        if row == 0:
            I += col
            col = 0
        elif col == 0:
            D += row
            row = 0
        elif dist[row][col] == dist[row-1][col] + deletes:
            D += 1
            row = row-1
        elif dist[row][col] == dist[row][col-1] + inserts:
            I += 1
            col = col-1
        elif dist[row][col] == dist[row-1][col-1] + substitutes:
            S += 1
            row, col = row-1, col-1
        else:
            H += 1
            row, col = row-1, col-1
    D, I = I, D
    return H, D, S, I

def compute_acc(preds, labels, costs=(7, 7, 10)):
    # cost according to HTK: http://www.ee.columbia.edu/~dpwe/LabROSA/doc/HTKBook21/node142.html

    if not len(preds) == len(labels):
        raise ValueError('# predictions not equal to # labels')
    Ns, Ds, Ss, Is = 0, 0, 0, 0
    for i, _ in enumerate(preds):
        H, D, S, I = iterative_levenshtein(preds[i], labels[i], costs)
        Ns += len(labels[i])
        Ds += D
        Ss += S
        Is += I
    try:
        acc = 100*(Ns-Ds-Ss-Is)/Ns
    except ZeroDivisionError as err:
        raise ZeroDivisionError('Empty labels')
    return acc
