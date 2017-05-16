import numpy as np
from utils import read_file, make_response_set, hash_response_set

"""
input file format:

ans1, ans2, ..., ans_{numWorkers}
"""

class binary_handler:
	def __init__(self, numWorkers, numTasks, ratings_path):
		"""
		 p  = [	true neg, false neg(e1)

		        false pos(e0), true pos]

		self.sorted_bucket: hold all the bucket index sorted by their response set.
		the first one has the least 0 as response and the last one has the most 0 as 
		response

		self.bucket2i: bucket_id --> list of items that in the bucket
		self.current_truths: list of labels for each task under current mapping function
		self.m2bucket: response set --> bucket number
		"""

		self.numWorkers = numWorkers
		self.numTasks = numTasks
		
		self.f = {}
		self.data = read_file(ratings_path)
		for i in xrange(self.numTasks):
			# if np.mean(self.data[i]) > 0.5:
			if np.random.rand() > 0.5:
				self.f[i] = 1
			else:
				self.f[i] = 0

		self.p = np.array([[0.6,0.4],[0.4,0.6]])
		
		self.current_truths = np.array(map(self.mapping_func, range(numTasks)))
		self.max_iter = max_iter
		self.bucket2i = {}
		self.m2bucket = {}
		self.sorted_bucket = []

	def mapping_func(self, key):
		return self.f[key]

	def estimate_e0e1(self):
		"e0 is false positive, e1 is false negative"
		# get counts for all positive and negative samples
		pos = np.sum(self.current_truths == 1) * self.numWorkers
		neg = np.sum(self.current_truths == 0) * self.numWorkers
		# get false positive and false negative counts
		false_neg = np.sum(self.data[self.current_truths == 1] == 0)
		false_pos = np.sum(self.data[self.current_truths == 0] == 1)
		#get values
		if neg == 0:
			self.p[1,0] = 0
		else:
			self.p[1,0] = float(false_pos)/neg
		if pos == 0:
			self.p[0,1] = 0
		else:
			self.p[0,1] = float(false_neg)/pos
		
		self.p[0,0] = 1 - self.p[1,0]
		self.p[1,1] = 1 - self.p[0,1]

	def bucketize(self):
		# get all the response sets
		Ms = make_response_set(self.data, 2)
		# print Ms
		#hash them, update dictionaries
		hash_response_set(Ms,self.bucket2i,self.m2bucket)
		#sanity check
		for bucket_id in self.bucket2i:
			for item_id in self.bucket2i[bucket_id]:
				m = tuple(Ms[item_id])
				assert self.m2bucket[m] == bucket_id, "bucket and item are inconsistent"

	def dominance_sort(self):
		# get sorted buckets
		temp = sorted(self.m2bucket.items(), key=lambda (response, bucket_id): response)
		self.sorted_bucket = [stuff[1] for stuff in temp]

	def compute_likelihood(self):
		# get the four probability
		true_pos = self.p[1,1]
		true_neg = self.p[0,0]
		false_pos = self.p[1,0]
		false_neg = self.p[0,1]

		# get counts for different categories: TP TN FP FN
		true_pos_count = np.sum(self.data[self.current_truths == 1] == 1)
		true_neg_count = np.sum(self.data[self.current_truths == 0] == 0)
		false_pos_count = np.sum(self.data[self.current_truths == 0] == 1)
		false_neg_count = np.sum(self.data[self.current_truths == 1] == 0)

		assert true_pos_count+true_neg_count+false_pos_count\
		+false_neg_count == self.numWorkers*self.numTasks, "counts and data are inconsistent"
		# compute log likelihood
		likelihood = np.log(true_pos+10**(-9))*true_pos_count+np.log(true_neg+10**(-9))*true_neg_count\
					+np.log(false_pos+10**(-9))*false_pos_count+np.log(false_neg+10**(-9))*false_neg_count

		return likelihood

	def update_f(self):

		max_lh = -1000000000
		max_f = {}
		count = 0
		for cut_point in xrange(1,len(self.sorted_bucket)-1):
			# get all the buckets containing items that should be 1
			one_bucket_indices = self.sorted_bucket[:cut_point]
			# get all the buckets containing items that should be 0
			zero_bucket_indices = self.sorted_bucket[cut_point:]
			#update mapping function
			for bucket in one_bucket_indices:
				for item in self.bucket2i[bucket]:
					self.f[item] = 1

			for bucket in zero_bucket_indices:
				for item in self.bucket2i[bucket]:
					self.f[item] = 0
			#update current prediction
			self.current_truths = np.array(map(self.mapping_func, range(self.numTasks)))
			self.estimate_e0e1()
			#compute likelihood under current setting
			temp_lh = self.compute_likelihood()
			#if the new mapping function produces larger likelihood, replace the old one
			if temp_lh > max_lh:
				max_lh = temp_lh
				max_f = self.f.copy()
		#update f
		self.f = max_f

	def train(self):
		it = 0
		print "begin bucketizing..."
		self.bucketize()
		print "sorting..."
		self.dominance_sort()
		print "updating..."
		self.update_f()
		print "final results:"
		print self.f.values()

		print "worker errors distribution: "
		print  self.p

	def eval(self):
		estimation = np.array(self.f.values())
		groundtruth = np.loadtxt("IC_data/IC_Gold.txt")

		print "accuracy: ", np.sum(estimation == groundtruth)/float(self.numTasks)   

if __name__ == "__main__":
	bh = binary_handler(19, 48, "IC_data/IC_Data.txt")
	bh.train()
	bh.eval()

