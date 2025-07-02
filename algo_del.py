a = [1, 2, 3, 5]

def uniq_order(seq):

    final = []
    final.append(seq[0])
    for i in range(1,len(seq)):
        if seq[i] == seq[i-1]:
            continue
        else:
            final.append(seq[i])
    return final




print(uniq_order(a))