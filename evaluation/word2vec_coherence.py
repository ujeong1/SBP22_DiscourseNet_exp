import inter_coherence
import intra_coherence
filename1 = "ship"
t_i = intra_coherence.score(filename1)
filename2 = "earn"
t_j = intra_coherence.score(filename2)
t_ij = inter_coherence.score(filename1, filename2)
result = ((t_i+t_j)/2)/t_ij
print(result)
