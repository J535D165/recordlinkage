# measures.py

import numpy

def true_positives(true_match_index, matches_index):
	""" The number of true positives.

	Return the number of correctly classified links.

	:param true_match_index: The golden/true links.  
	:param match_index: The classified links.  

	:return: The number of correctly classified links. 
	:rtype: int
	"""	

	return len(true_match_index & matches_index)

def true_negatives(true_match_index, matches_index, n_pairs):
	""" The number of true negatives.

	Return the number of correctly classified non-links.

	:param true_match_index: The golden/true links.  
	:param match_index: The classified links.  

	:return: The number of correctly classified non-links. 
	:rtype: int
	"""	

	if type(n_pairs) != int:
		n_pairs = len(n_pairs)

	return n_pairs - len(true_match_index | matches_index)

def false_positives(true_match_index, matches_index):
	""" The number of false positives.

	:param true_match_index: The golden/true links.  
	:param match_index: The classified links.  

	:return: 
	:rtype: int
	"""	

	# The classified matches without the true matches.
	return len(matches_index - true_match_index)

def false_negatives(true_match_index, matches_index):
	""" The number of false negatives.

	:param true_match_index: The golden/true links.  
	:param match_index: The classified links.  

	:return: 
	:rtype: int
	"""	
	return len(true_match_index - matches_index)

def confusion_matrix(true_match_index, matches_index, pairs):
	""" Return the confusion matrix.

	The confusion matrix is used to compute measures like precision and recall.

	:param true_match_index: The golden/true links.  
	:param match_index: The classified links.  

	:return: 
	:rtype: numpy.array
	"""	

	# True positives
	tp = true_positives(true_match_index, matches_index)

	# True negatives
	tn = true_negatives(true_match_index, matches_index, pairs)

	# False positives
	fp = false_positives(true_match_index, matches_index)

	# False negatives
	fn = false_negatives(true_match_index, matches_index)

	return numpy.array([[tp, fn], [fp, tn]])

def precision(confusion_matrix):
	""" Compute the precision

	The precision is given by tp/(tp+fp).

	:param confusion_matrix: The matrix with tp, fn, fp, tn values.  

	:return: The precision 
	:rtype: float
	"""

	return float(confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[1,0]))

def recall(confusion_matrix):
	""" Compute the recall/sensitivity

	The recall is given by tp/(tp+fn).

	:param confusion_matrix: The matrix with tp, fn, fp, tn values.  

	:return: The recall 
	:rtype: float
	"""

	return float(confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[0,1]))

def accuracy(confusion_matrix):
	""" Compute the accuracy

	The accuracy is given by (tp+tn)/(tp+fp+tn+fn).

	:param confusion_matrix: The matrix with tp, fn, fp, tn values.  

	:return: The accuracy 
	:rtype: float
	"""

	return float((confusion_matrix[0,0]+confusion_matrix[1,1])/numpy.sum(confusion_matrix))

def specificity(confusion_matrix):
	""" Compute the specitivity

	The specitivity is given by tn/(fp+tn).

	:param confusion_matrix: The matrix with tp, fn, fp, tn values.  

	:return: The accuracy 
	:rtype: float
	"""

	return float(confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[1,0]))

def fscore(confusion_matrix):
	""" Compute the f_score

	The fscore is given by 2*(precision*recall)/(precision+recall).

	:note: If there are no pairs classified as links, this measure will raise a ZeroDivisionError.

	:param confusion_matrix: The matrix with tp, fn, fp, tn values.  

	:return: The fscore 
	:rtype: float
	"""

	prec = precision(confusion_matrix)
	rec = recall(confusion_matrix)

	return float(2*prec*rec/(prec+rec))



