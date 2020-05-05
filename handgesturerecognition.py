#%%
import numpy as np
import glob

test_names = []
labels = []
movements = {'cyl':[],'hook':[],'lat':[],'palm':[],'spher':[],'tip':[]}
label = 1
for movement in movements:
    skip_count = 0
    for f in glob.glob("ALL FEATURES/Vini Data/"+movement+"/*.csv"):
        skip_count +=1
        if (skip_count <= 6):
            test_names.append(f)
        if (skip_count > 6):
            movements[movement].append(np.genfromtxt(f,delimiter=","))
    movements[movement] = np.concatenate(movements[movement],axis=0)
    labels.append(label*np.ones(len(movements[movement])))
    label+=1
# %%
def kMeans(data,n,tol,max_iter):
    rnd_idx = np.array(range(len(data)))
    np.random.shuffle(rnd_idx)
    centroids = data[rnd_idx[:n]]
    cluster_for_point = np.zeros(len(data))
    for it in range(max_iter):
        done = True
        clusters = [[] for i in range(n)]
        for i,point in enumerate(data):
            distances = [np.linalg.norm(point-centroid) for centroid in centroids]
            your_cluster_idx = distances.index(min(distances))
            clusters[your_cluster_idx].append(point)
            cluster_for_point[i] = your_cluster_idx
        old_centroids = centroids
        for i,cluster in enumerate(clusters):
            centroids[i] = np.mean(cluster, axis=0)
        for i,centroid in enumerate(centroids):
            if (np.sum((centroid - old_centroids[i] / old_centroids[i]) * 100.0) > tol):
                done = False
        if(done):
            break
    return centroids,cluster_for_point

from sklearn.cluster import KMeans

all_movements = []
for movement in movements:
    movements[movement] = np.array(movements[movement])
    all_movements.append(movements[movement])


all_movements = np.concatenate(all_movements,axis=0)
labels = np.concatenate(labels,axis=0)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=10).fit(all_movements,labels)
all_movements = lda.transform(all_movements)

km = KMeans(n_clusters=75).fit(all_movements)
O_centroids = km.cluster_centers_
O_movements = km.labels_

#O_centroids,O_movements = kMeans(all_movements[:,1:],75,0.0001,300)

#%%
O_cyl = O_movements[:movements['cyl'].shape[0]]
i = len(O_cyl)
O_hook = O_movements[i:i+movements['hook'].shape[0]]
i = i + len(O_hook)
O_lat = O_movements[i:i+movements['lat'].shape[0]]
i = i + len(O_lat)
O_palm = O_movements[i:i+movements['palm'].shape[0]]
i = i + len(O_palm)
O_spher = O_movements[i:i+movements['spher'].shape[0]]
i = i + len(O_spher)
O_tip = O_movements[i:i+movements['tip'].shape[0]]
#%%
import sys

sys.setrecursionlimit(10**6) 

num_states = 20
num_sym = len(set(O_movements))

Os = {'cyl':O_cyl,'hook':O_hook,'lat':O_lat,'palm':O_palm,'spher':O_spher,'tip':O_tip,}

for movement in movements:
    Os[movement] = Os[movement].astype(int)
# Transition Probabilities
As = {'cyl':np.array([]),'hook':np.array([]),'lat':np.array([]),'palm':np.array([]),'spher':np.array([]),'tip':np.array([])}
for A in As:
    As[A] = np.random.random(size=(num_states,num_states))
    As[A] = As[A] / np.sum(As[A] , axis=1)


# Emission Probabilities
Bs = {'cyl':np.array([]),'hook':np.array([]),'lat':np.array([]),'palm':np.array([]),'spher':np.array([]),'tip':np.array([])}
for B in Bs:
    Bs[B] = np.random.random(size=(num_states,num_sym))
    Bs[B] = Bs[B] / np.sum(Bs[B], axis=1).reshape((-1,1))

# Equal Probabilities for the initial distribution
PI= np.zeros(num_states)
PI[0] = 1

#%%
def forward(o,a,b,pi,t,alpha,alpha_div):
    b[b==0] = 10**-10
    if (t==0):
        alpha[t] = pi * b[:,o[t]]
    else:
        alpha[t] = forward(o,a,b,pi,t-1,alpha,alpha_div)[0][t-1].dot(a) * b[:,o[t]]
    alpha_div[t] = np.repeat(np.sum(alpha[t]).reshape(1,1),len(alpha[t]),axis=1)
    alpha[t] = alpha[t] / alpha_div[t]
    #alpha_div[t][alpha_div[t] == 0] = 1
    #alpha = alpha/alpha_div
    return alpha,alpha_div

def backward(o,a,b,t,beta,c):
    b[b==0] = 10**-10
    if (t==len(beta)-1):
        beta[t] = np.ones(beta.shape[1]) / c[t]
    else:
        beta[t] = ((backward(o,a,b,t+1,beta,c)[t+1] * b[:,o[t+1]]).dot(a.T)) / c[t]
    return beta

def baumWelch(o,a,b,pi,n_iter):
    T = len(o)
    N = len(a)
    alpha_scales = np.zeros((n_iter, len(o), len(a)))
    for i in range(n_iter):
        b[b==0] = 10**-10
        alpha = np.zeros((len(o), len(a)))
        beta = np.zeros((len(o), len(a)))

        #E-Step
        alpha,alpha_scales[i] = forward(o,a,b,pi,T-1,alpha,alpha_scales[i])

        beta = backward(o,a,b,0,beta,alpha_scales[i])
        
        xi = np.zeros((N,N,T-1))

        for t in range(T-1):
            for i in range(N):
                xi[i,:,t] = (alpha[t,i] * a[i, :] * b[:,o[t+1]] * beta[t+1,:]) / \
                alpha[t].dot((beta[t+1] * b[:,o[t+1]]).dot(a.T))

        gamma = np.sum(xi, axis=1)
        
        #M-Step
        a = np.sum(xi,axis=2) / np.sum(gamma,axis=1).reshape((-1, 1))

        gamma = np.concatenate((gamma, np.sum(xi[:, :, T-2], axis=1).reshape((-1, 1))), axis=1)

        for vk in range(num_sym):
            b[:,vk] = np.sum(gamma[:,o == vk],axis=1)

        b = b / np.sum(gamma, axis=1).reshape((-1,1))

    return a,b,alpha_scales[:,:,0]

cs = {'cyl':np.array([]),'hook':np.array([]),'lat':np.array([]),'palm':np.array([]),'spher':np.array([]),'tip':np.array([])}
num_iter = 4
for movement in movements:
    As[movement], Bs[movement], cs[movement] = baumWelch(Os[movement],As[movement],Bs[movement],PI,num_iter)
#%%
import matplotlib.pyplot as plt

for movement in movements:
    log_like = [-np.sum(np.log(1/c)) for c in cs[movement]]
    plt.plot(range(1,num_iter+1),log_like)

plt.legend(movements)
plt.title("Log Likelihood Plot Over Iterations")
plt.xlabel("Iterations")
plt.ylabel("Log Likelihood")

plt.show()
# %% 
import time
temp_names = [test_names[24]]
#t = time.time()
#wrong = []
#wrong_scores = []
for w in range(23):
    preds = []
    trues = []
    correct_count = 0
    #test_names = [f for f in glob.glob("female_1/"+movement+"/*.csv") for movement in movements]
    for test_name in temp_names:
        raw_obs = np.genfromtxt(test_name,delimiter=",")
        obs = km.predict(lda.transform(raw_obs[4*w:4*(w+1),:]))
        log_p = {'cyl':0,'hook':0,'lat':0,'palm':0,'spher':0,'tip':0}
        for movement in movements:
            new_alpha = np.zeros((len(obs), len(As[movement])))
            c_t = np.zeros((len(obs), len(As[movement])))
            new_alpha,c_t = forward(obs,As[movement],Bs[movement],PI,len(obs)-1,new_alpha,c_t)
            log_p[movement] = (-np.sum(np.log(1/c_t[:,0])))
        real_test_class = test_name.split("/")[2]
        pred_test_class = max(log_p, key=log_p.get)
        trues.append(real_test_class)
        preds.append(pred_test_class)
        if (real_test_class == pred_test_class):
            correct_count += 1
        #else:
            #wrong.append((test_name,pred_test_class))
            #wrong_scores.append(log_p)
#elapsed = time.time() - t

    print('[Window %d]: Accuracy: %.1f%%' % (w,(correct_count/len(temp_names))*100))
    print(pred_test_class)
#print("It took " + str(elapsed) + " seconds to classify " + str(len(test_names)) + " examples.")
#%%
with open('preds.txt', 'w') as f:
    for l in preds:
        f.write('%s\n' % l)

with open('trues.txt', 'w') as f:
    for l in trues:
        f.write('%s\n' % l)
#%%
centers = km.cluster_centers_
np.savetxt("centers.csv",centers,delimiter=',')
for movement in movements:
    np.savetxt("A"+movement+".csv", As[movement], delimiter=",")
    np.savetxt("B"+movement+".csv", Bs[movement], delimiter=",")
# %%
