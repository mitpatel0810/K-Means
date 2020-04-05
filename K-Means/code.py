from scipy.io import loadmat
#import matplotlib.pyplot as plt
import numpy as np

def euclideanDistance(x,y):

	return np.sqrt(np.sum((x-y)**2))

class KMeans():

	def __init__(self, k=2, iter=100, strategy=1):

		self.k = k
		self.iter = iter
		self.strategy = strategy

		#to store index of each cluser
		self.clusters = [[] for _ in range(self.k)]

		#to store centroid value for each cluster
		self.centroids = []

		#value of objective function
		self.sse = 0

	def predict(self, data):

		self.data = data
		self.samples, self.dimensions = data.shape
		
		if self.strategy == 1:
			#print("In strategy 1")

			#initialize centroid at any random point from given data
			idx = np.random.choice(self.samples, self.k, replace=False)
			
			self.centroids = [self.data[i] for i in idx]
			

		else:
			#sprint("In Strategy 2")
			self.dummyData = self.data
			#print(len(self.dummyData), len(self.data))

			#initialise first center at any random point
			first_idx = np.random.choice(self.samples, 1)
			
			tmp = []
			for d in self.data[first_idx]:
				for k in d:
					tmp.append(k)
			
			
			self.centroids.append(tmp)
		

			self.dummyData = np.delete(self.dummyData, first_idx, 0)
			#print(len(self.dummyData))

			#get other centroids as far as possible from each other
			for i in range(2,self.k+1):

				dist = np.empty(len(self.dummyData))
					

				for i in range(len(self.dummyData)):
					d = 0
					for c in self.centroids:
						d = d + euclideanDistance(self.dummyData[i],c)

					dist[i] = d
				# print(max(dist))
				max_idx = np.argmax(dist)
				# print(max_idx,dist[max_idx])
				# print(self.dummyData[max_idx])
				temp = []
				for d in self.dummyData[max_idx]:
					temp.append(d)

				self.centroids.append(temp)
				self.dummyData = np.delete(self.dummyData,max_idx,0)

		#print(self.centroids)
		
		#Now optimize cluster by improving centroids at every iteration
		for i in range(self.iter):

			#generate cluster indexes by assigning all the datapoints to nearest centroids
			self.clusters = self._generateCluster(self.centroids)
			

			#get new centroids based on the returned clusters

			#self.plot()

			old_centroid = self.centroids
			self.centroids = self._getUpdatedCentroid(self.clusters)

			#now check if algorithm has converged
			if sum(euclideanDistance(old_centroid[i],self.centroids[i]) for i in range(self.k)) ==0:
				#print(i)
				break

			#print(self.clusters)

			#self.plot()

		#calculate objective function for this cluster
		for i in range(self.k):
			#print(self.clusters[i])

			self.sse = self.sse + self._calculateSSE(self.clusters[i], i)
		#self.plot()
	
		return self._getClusterLabel(self.clusters)



	def _generateCluster(self, centroids):

		#assign every datapoint to closest centroids
		clusters = [[] for _ in range(self.k)]

		for i in range(self.samples):

			centroid_index = self._getClosestCentroid(self.data[i], centroids)
			clusters[centroid_index].append(i)

		return clusters


	def _getClosestCentroid(self, sample, centroids):

		dist = [euclideanDistance(sample, point) for point in centroids]

		#get index of closest centroid

		closest_index = np.argmin(dist)
		return closest_index

	def _getUpdatedCentroid(self, clusters):

		#initialize centroids with zeros
		centroids = np.zeros((self.k, self.dimensions))

		for clusterNumber, clusteredDataIndex in enumerate(clusters):

			#take mean of each cluster's data and assign it to new centroid
			#print(self.data[clusteredDataIndex])
			cluster_mean = np.mean(self.data[clusteredDataIndex], axis=0)
			centroids[clusterNumber] = cluster_mean

		return centroids


	def _getClusterLabel(self, clusters):

		#initialise label array to assign the cluster number for all data points
		labels = np.zeros(self.samples)

		for i in range(len(clusters)):
			#for each cluster number
			for clusteredDataIndex in clusters[i]:

				#assign cluster number to data index
				labels[clusteredDataIndex] = i

		return labels

	
	def _calculateSSE(self, array, indx):

		temp = []

		for i in array:
			temp.append(self.data[i])

		
		sum = 0

		for i in temp:
			sum = sum + np.sum((i-self.centroids[indx])**2)

		return sum




	def plot(self):
		

		fig,ax = plt.subplots(figsize=(12,8))

		for i, index in enumerate(self.clusters):

			point = self.data[index].T
			ax.scatter(*point)

		#print(self.centroids)

		for point in self.centroids:

			ax.scatter(*point, marker="*", color="black", linewidth=2)

		plt.title('Strategy: {}'.format(self.strategy))
		plt.xlabel("Number of Clusters: {}".format(self.k))
		plt.show()


if __name__ == '__main__':

	data = loadmat('AllSamples.mat')

	data = data['AllSamples']

	sse1 = []
	sse2 = []
	k_values = []

	for i in range(2,11):
		k_values.append(i)

	for i in k_values:
		

		k1 = KMeans(k=i, iter=100, strategy=1)
		k2 = KMeans(k=i, iter=100, strategy=2)

		result1 = k1.predict(data)
		result2 = k2.predict(data)
		# print("K values: ",i)
		# print("Result: ",result)
		#print(k.sse)
		sse1.append(k1.sse)
		sse2.append(k2.sse)

	#print(len(result))
	
	#sse = [i/max(sse) for i in sse]
	#print(sse)
	# fig = plt.figure()
	# ax = fig.add_subplot(121)
	# ax.plot(k_values,sse2)
	# plt.suptitle('Plot for Strategy 2')
	# plt.xlabel('Number of Clusters')
	# plt.ylabel('Objective Function')
	# #ax.legend(loc='best')
	# fig.show()
	# plt.show()

	

	# x = data[:,0]
	# y = data[:,1]

	# plt.scatter(x,y)
	# plt.xlabel('X-axis')
	# plt.ylabel('Y-axis')
	# plt.show()

