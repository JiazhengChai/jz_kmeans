import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self,n_clusters=3, n_runs=10, max_iter=300, tol=0.0001,debug=False):
        self.n_clusters=n_clusters
        self.n_runs=n_runs
        self.max_iter=max_iter
        self.tol=tol
        self.debug=debug

        self.cluster_centers_=None
        self.labels_=None
        self.fitted=False

    def fit(self,X):
        assert len(X[0])==2
        assert self.n_clusters<=len(X)
        mem=defaultdict()

        ###loop for all trials
        for trial in range(self.n_runs):
            if self.debug:
                print(" ")
                print("Trials: ", trial)

            ###loop for one trial of Kmeans
            indices=np.random.choice(len(X),size=self.n_clusters,replace=False)
            cur_centers=[X[i] for i in indices]

            converged=False
            it=0
            while not converged:
                cur_labels=self._assign_labels(X,cur_centers)
                prev_centers=cur_centers
                cur_centers=self._update_centers(X,cur_labels)
                if self._delta_prev_cur_centers(prev_centers,cur_centers)<self.tol:
                    converged=True

                it+=1
                if it==self.max_iter:
                    break

                if self.debug:
                    print("iteration: ",it)
                    print("centers: ",cur_centers)

            if self.debug:
                print(" ")

            mem[trial]=[cur_centers,cur_labels]

        self.cluster_centers_,self.labels_=self._return_best(X,mem)
        self.fitted=True

    def predict(self,X):
        assert len(X[0]) == 2
        if not self.fitted:
            print("Model not fitted. Please fit your model first before predicting.")
            return

        return self._assign_labels(X,self.cluster_centers_)

    def _assign_labels(self,X,cur_centers):
        cur_labels=[]
        for point in X:
            min_dist_ind=-1
            min_dist=-1
            for i,c in enumerate(cur_centers):
                if min_dist<0:
                    min_dist=self._calcDist(point,c)
                    min_dist_ind=i
                else:
                    tmp=self._calcDist(point,c)
                    if tmp<min_dist:
                        min_dist=tmp
                        min_dist_ind=i

            cur_labels.append(min_dist_ind)

        assert len(cur_labels)==len(X)

        return cur_labels

    def _update_centers(self,X,cur_labels):
        new_centers=[[0,0] for i in range(self.n_clusters)]
        count_per_class=[0 for i in range(self.n_clusters)]

        for i,point in enumerate(X):
            count_per_class[cur_labels[i]]+=1

            new_centers[cur_labels[i]][0]+=point[0]
            new_centers[cur_labels[i]][1]+=point[1]

        for i in range(len(new_centers)):
            new_centers[i][0]/=count_per_class[i]
            new_centers[i][1]/=count_per_class[i]
            new_centers[i]=tuple(new_centers[i])

        assert len(new_centers)==self.n_clusters
        return new_centers

    def _delta_prev_cur_centers(self,old_centers,new_centers):
        delta=0
        for i in range(len(old_centers)):
            old_c=old_centers[i]
            new_c=new_centers[i]
            delta+=np.sqrt(self._calcDist(old_c,new_c))
        return delta

    def _calcDist(self,p1,p2):
        return (p1[0]-p2[0])**2+(p1[1]-p2[1])**2

    def _return_best(self,X,all_trials_results):
        best_centers=[]
        best_labels=[]
        min_var=-1
        for i in range(self.n_runs):
            cur_centers,cur_labels=all_trials_results[i]
            variance_by_sum_of_distance=0

            for j,point in enumerate(X):
                variance_by_sum_of_distance+=self._calcDist(point,cur_centers[cur_labels[j]])

            if self.debug:
                print("Variance distance of trial {} is {}".format(i,variance_by_sum_of_distance) )

            if min_var<0:
                min_var=variance_by_sum_of_distance
                best_centers=cur_centers
                best_labels=cur_labels
            else:
                if variance_by_sum_of_distance<min_var:
                    min_var = variance_by_sum_of_distance
                    best_centers = cur_centers
                    best_labels = cur_labels

        #if self.debug:
        print(" ")
        print("Best Variance distance: {}".format(min_var))
        print("Best Centers: ")
        print(best_centers)
        print("Best Labels: ")
        print(best_labels)
        print(" ")

        return best_centers,best_labels

if __name__=="__main__":
    kmeans=KMeans(n_clusters=5,n_runs=10,debug=False)

    X1=np.random.normal(0,1,size=(100,2))
    X2=np.random.normal(0,1,size=(100,2))

    kmeans.fit(X1)

    predicted_labels=kmeans.predict(X2)
    plt.figure()
    plt.title("Train data")
    plt.scatter(X1[:, 0], X1[:, 1], c=kmeans.labels_)
    for c in kmeans.cluster_centers_:
        plt.scatter(c[0], c[1], c="red", marker="x")
    plt.xlim([-5,5])
    plt.ylim([-5,5])
    plt.savefig("train.jpg")

    plt.figure()
    plt.title("Test data")
    plt.scatter(X2[:,0],X2[:,1],c=predicted_labels)#kmeans.labels_
    for c in kmeans.cluster_centers_:
        plt.scatter(c[0],c[1],c="red",marker="x")
    plt.xlim([-5,5])
    plt.ylim([-5,5])
    plt.savefig("test.jpg")

    plt.show()

