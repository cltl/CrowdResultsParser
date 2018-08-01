import numpy as np
from scipy.stats import entropy
from collections import defaultdict

def replace_zeros(data):
    """
    Replace all zeros with very small values.
    """
    new_value = 0.0000000001
    new_data = []
    for value in data:
        if value>0:
            new_data.append(value)
        else:
            new_data.append(new_value)
    return new_data

def js(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
   # normalize
    p /= p.sum()
    q /= q.sum()
    m = (p + q) / 2
    return (entropy(p, m) + entropy(q, m)) / 2

def kl(p, q):
    """Kullback-Leibler divergence D(P || Q) for discrete distributions
    Parameters
    ----------
    p, q : array-like, dtype=float, shape=n
    Discrete probability distributions.
    """
    p = replace_zeros(p)
    q = replace_zeros(q)
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)

    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def compute_property_divergence(domain_values, gold_values, sys_values):
    s=[]
    g=[]
    for a_value in domain_values:
        if a_value in sys_values.keys():
            s.append(sys_values[a_value])
        else:
            s.append(0.0)
        if a_value in gold_values.keys():
            g.append(gold_values[a_value])
        else:
            g.append(0.0)
    print(s,g)
    print(js(s,g))
    return js(s,g)

def evaluate_all_rows(domain, gold_data, sys_data):
    divergencies = defaultdict(list)

    for index, gold_row in enumerate(gold_data):
        sys_row=sys_data[index]
        print(sys_row)
        print(gold_row)
        for property, values in gold_row.items():
            domain_values = domain[property]
            sys_prop_values = sys_row[property]
            gold_prop_values = gold_row[property]

            div = compute_property_divergence(domain_values, gold_prop_values, sys_prop_values)
            divergencies[property].append(div)

    return divergencies
    

if __name__ == "__main__":
    a = [0.8, 0.2, 0.0]
    b = [0.0, 1.0, 0.0]
    c = [0.0, 0.0, 1.0]
    print(kl(c,a))
    print(entropy(replace_zeros(c), replace_zeros(a)))
    print(js(a,c))
    print(js(c,a))

    domain = {'gender': ['male', 'female'], 'political party': ['Republican party', 'Democratic party']}

    sys_data=[{'gender': {'male': 0.8, 'female': 0.2}, 'political party': {'Republican party': 0.8, 'Democratic party': 0.2}}, {'gender': {'female': 0.5, 'male': 0.5}, 'political party': {'Republican party': 0.8, 'Democratic party': 0.2}}, {'gender': {'male': 1.0}, 'political party': {'Republican party': 0.8, 'Democratic party': 0.2}}]
    gold_data=[{'gender': {'male': 1.0}, 'political party': {'Republican party': 0.5, 'Democratic party': 0.5}}, {'gender': {'male': 0.6, 'female': 0.4}, 'political party': {'Republican party': 0.3, 'Democratic party': 0.7}}, {'gender': {'female': 1.0}, 'political party': {'Republican party': 0.5, 'Democratic party': 0.3}}]

    divergencies = evaluate_all_rows(domain, gold_data, sys_data)
    print(divergencies)

    for property, divs in divergencies.items():
        print(property, sum(divs)/len(divs))
