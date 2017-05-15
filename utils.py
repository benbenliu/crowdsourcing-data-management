import numpy as np

def read_file(path):
	return np.loadtxt(path)

def make_response_set(data, num_class):
	"""
	given response data for each task, compute response set
	"""
	def count_number(a, num_class):
		res = np.zeros(num_class)
		for i in xrange(num_class):
			res[i] = np.sum(a==i)
		return res

	return np.apply_along_axis(count_number, 1, data, num_class)

def hash_response_set(Ms, bucket2i, m2bucket):
	"""
	Ms: response set of all the tasks
	bucket: as the name refers, dictionary
	"""
	bucket_index = 0
	for i in xrange(Ms.shape[0]):
		#i is the index for task
		m = tuple(Ms[i])
		if m not in m2bucket:
			bucket2i[bucket_index] = [i]
			m2bucket[m] = bucket_index
			bucket_index+=1
		else:
			bucket2i[m2bucket[m]].append(i)
			
# test = np.array([[0,0,1],[1,1,2], [2,2,3]])
# print make_response_set(test,4)

