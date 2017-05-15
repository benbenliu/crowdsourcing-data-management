import numpy as np
from utils import read_file, make_response_set, hash_response_set

"""
input file format:

ans1, ans2, ..., ans_{numWorkers}
"""

class binary_handler:
	def __init__(self, numWorkers, numTasks, ratings_path, max_iter = 100):
		"""
		 p  = [	true neg, false neg(e1)

		        false pos(e0), true pos]

		sorted_bucket: hold all the bucket index sorted by their response set.
		the first one has the least 0 as response and the last one has the most 0 as 
		response

		bucket2i: bucket_id
		"""

		self.numWorkers = numWorkers
		self.numTasks = numTasks
		
		self.f = {}
		for i in xrange(self.numTasks):
			self.f[i] = 0

		self.p = np.array([[0.6,0.4],[0.4,0.6]])
		self.data = read_file(ratings_path)
		self.current_truths = np.array(map(self.mapping_func, range(numTasks)))
		self.max_iter = max_iter
		self.bucket2i = {}
		self.m2bucket = {}
		self.sorted_bucket = []

	def mapping_func(self, key):
		return self.f[key]

	def estimate_e0e1(self):
		"e0 is false positive, e1 is false negative"
		pos = np.sum(self.current_truths == 1) * self.numWorkers
		neg = np.sum(self.current_truths == 0) * self.numWorkers

		false_neg = np.sum(self.data[self.current_truths == 1] == 0)
		false_pos = np.sum(self.data[self.current_truths == 0] == 1)

		self.p[1,0] = float(false_pos)/neg
		self.p[0,1] = float(false_neg)/pos
		self.p[0,0] = 1 - self.p[1,0]
		self.p[1,1] = 1 - self.p[0,1]

	def bucketize(self):
		Ms = make_response_set(self.data, 2)
		hash_response_set(Ms,self.bucket2i,self.m2bucket)
		
		#sanity check
		for bucket_id in self.bucket2i:
			for item_id in self.bucket2i[bucket_id]:
				m = tuple(Ms[item_id])
				assert self.m2bucket[m] == bucket_id, "bucket and item are inconsistent"

	def dominance_sort(self):
		temp = sorted(self.m2bucket.items(), key=lambda (response, bucket_id): response)
		self.sorted_bucket = [stuff[1] for stuff in temp]

	def compute_likelihood(self):
		true_pos = self.p[1,1]
		true_neg = self.p[0,0]
		false_pos = self.p[1,0]
		false_neg = self.p[0,1]

		true_pos_count = np.sum(self.data[self.current_truths == 1] == 1)
		true_neg_count = np.sum(self.data[self.current_truths == 0] == 0)
		false_pos_count = np.sum(self.data[self.current_truths == 0] == 1)
		false_neg_count = np.sum(self.data[self.current_truths == 1] == 0)

		assert true_pos_count+true_neg_count+false_pos_count\
		+false_neg_count == self.numWorkers*self.numTasks, "counts and data are inconsistent"

		likelihood = np.log(true_pos)*true_pos_count+np.log(true_neg)*true_neg_count\
					+np.log(false_pos)*false_pos_count+np.log(false_neg)*false_neg_count

		return likelihood

	def update_f(self):

		max_lh = 0
		max_f = {}

		for cut_point in xrange(len(self.sorted_bucket)):
			# get all the buckets containing items that should be 1
			one_bucket_indices = self.sorted_bucket[cut_point:]
			# get all the buckets containing items that should be 0
			zero_bucket_indices = self.sorted_bucket[:cut_point]
			#update mapping function
			for bucket in one_bucket_indices:
				for item in bucket2i[bucket]:
					self.f[item] = 1

			for bucket in zero_bucket_indices:
				for item in bucket2i[bucket]:
					self.f[item] = 0
			#update current prediction
			self.current_truths = np.array(map(self.mapping_func, range(self.numTasks)))
			#compute likelihood under current setting
			temp_lh = self.compute_likelihood()
			#if the new mapping function produces larger likelihood, replace the old one
			if temp_lh > max_lh:
				max_lh = temp_lh
				max_f = self.f
		#update
		self.f = max_f
	
	def train(self):
		it = 0
		self.bucketize()
		self.dominance_sort()

		while it < self.max_iter:
			self.estimate_e0e1()
			self.bh.update_f()

			it+=1


bh = binary_handler(19, 48, "IC_data/IC_Data.txt")
bh.train()





		






