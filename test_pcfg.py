import math
import pprint
import torch
import torch.nn as nn
import torch.nn.functional as F
from pcfg import PCFG

def p(s):
  print(s)
  return s

gen_a= lambda terminals_per_category: PCFG.fromstring( " A    -> "+ " | ".join(["\"a" + str(i) + "\" [" + str(1/terminals_per_category) + "]" for i in range(terminals_per_category) ]))

generate_test_grammar = lambda p1,terminals_per_category: PCFG.fromstring( " S  -> NP-s VP-s [" + str((1 - p1)/2) + "] | NP-s S VP-s [" + str(p1/2) + "] | NP-p VP-p [" + str((1 - p1)/2) + "] | NP-p S VP-p [" + str(p1/2) + "]\n" + 
                                                                           " NP-s -> N-s     [" + str(1) + "]\n" +
                                                                           " NP-p -> N-p     [" + str(1) + "]\n" +
                                                                           " VP-s -> V-s     [" + str(1) + "]\n" +
                                                                           " VP-p -> V-p     [" + str(1) + "]\n" +
                                                                          
                                                                           " N-s  -> "+ " | ".join(["\"n" + str(i) + "-s\" [" + str(1/terminals_per_category) + "]" for i in range(terminals_per_category) ]) + "\n" +
                                                                           " N-p  -> "+ " | ".join(["\"n" + str(i) + "-p\" [" + str(1/terminals_per_category) + "]" for i in range(terminals_per_category) ]) + "\n" +
                                                                           " V-s  -> "+ " | ".join(["\"v" + str(i) + "-s\" [" + str(1/terminals_per_category) + "]" for i in range(terminals_per_category) ]) + "\n" +
                                                                           " V-p  -> "+ " | ".join(["\"v" + str(i) + "-p\" [" + str(1/terminals_per_category) + "]" for i in range(terminals_per_category) ])  )
generate_grammar = lambda p1,p2,terminals_per_category: PCFG.fromstring( p(" S  -> NP-s VP-s [" + str((1 - p1)/2) + "] | NP-s S VP-s [" + str(p1/2) + "] | NP-p VP-p [" + str((1 - p1)/2) + "] | NP-p S VP-p [" + str(p1/2) + "]\n" + 
                                                                         " NP-s -> N-s     [" + str(1 - p2) + "] | A NP-s    [" + str(p2) + "]\n" +
                                                                         " NP-p -> N-p     [" + str(1 - p2) + "] | A NP-p    [" + str(p2) + "]\n" +
                                                                         " VP-s -> V-s     [" + str(1 - p2) + "] | A VP-s    [" + str(p2) + "]\n" +
                                                                         " VP-p -> V-p     [" + str(1 - p2) + "] | A VP-p    [" + str(p2) + "]\n" +
                                                                        
                                                                         " N-s  -> "+ " | ".join(["\"n" + str(i) + "-s\" [" + str(1/terminals_per_category) + "]" for i in range(terminals_per_category) ]) + "\n" +
                                                                         " N-p  -> "+ " | ".join(["\"n" + str(i) + "-p\" [" + str(1/terminals_per_category) + "]" for i in range(terminals_per_category) ]) + "\n" +
                                                                         " V-s  -> "+ " | ".join(["\"v" + str(i) + "-s\" [" + str(1/terminals_per_category) + "]" for i in range(terminals_per_category) ]) + "\n" +
                                                                         " V-p  -> "+ " | ".join(["\"v" + str(i) + "-p\" [" + str(1/terminals_per_category) + "]" for i in range(terminals_per_category) ]) + "\n" +
                                                                         " A    -> "+ " | ".join(["\"a" + str(i) + "\" [" + str(1/terminals_per_category) + "]" for i in range(terminals_per_category) ]) ))

gen_nest = lambda d,a,b: ([a for i in range(d)] + [b for i in range(d)])
gen_seq = lambda s,a: [a for i in range(s)]# + [b for i in range(d)])
gen_as=lambda n:[sent for sent in gen_a(5).generate(n)]
gen_inter_as=lambda d,s:[" ".join(gen_as(s)) for i in range(d*2 + 1)]

gen_depth = lambda d,n:[sent for sent in generate_test_grammar(.7,5).generate(n*10) if len(sent.split()) == d*2][:n]

def gen_test_set(d=3,s=2):
    test_set_size = 10
    content = [i.split() for i in gen_depth(d,test_set_size)]
    fillers =[gen_inter_as(d,s) for i in range(test_set_size)]
    test_set = [" ".join(i) for i in [" ".join([" ".join(gen_as(s))] +[val for pair in zip(content[i], fillers[i]) for val in pair]).split() for i in range(test_set_size)]]
    return test_set

terminals_per_category = 5
symbols = []
symbols.extend(["eos"])
symbols.extend(['n' + str(i) + '-s' for i in range(terminals_per_category)])
symbols.extend(['v' + str(i) + '-s' for i in range(terminals_per_category)])
symbols.extend(['n' + str(i) + '-p' for i in range(terminals_per_category)])
symbols.extend(['v' + str(i) + '-p' for i in range(terminals_per_category)])
symbols.extend(['a' + str(i) + '' for i in range(terminals_per_category)])

indx2tok = { i : symbols[i] for i in range(0, len(symbols) ) }
tok2indx = { symbols[i]:i  for i in range(0, len(symbols) ) }

test_sent_str = [sent for sent in generate_grammar(0.7, 0.7, 5).generate(1)]

print(test_sent_str)

gen_samples = lambda str_corpus, max_len: [([tok2indx[i] for i in (sent+ " eos").split() + ["eos"]*(max_len-len(sent.split())) ],
                                            [tok2indx[i] for i in (sent+ " eos").split() + ["eos"]*(max_len-len(sent.split()) + 1)][1:],
                                            [1]*(len(sent.split()))+[0]*(1+max_len-len(sent.split()))) 
                                            for sent in str_corpus]

print(gen_samples(test_sent_str, 6))
