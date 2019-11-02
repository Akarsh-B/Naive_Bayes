import re
import random
import math
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot") #Use GGPlot style for graph

random.seed(10)
"""
Read text data from file and pre-process text by doing the following
1. convert to lowercase
2. convert tabs to spaces
3. remove "non-word" characters
Store resulting "words" into an array
"""
FILENAME='SMSSpamCollection'
all_data = open(FILENAME).readlines()

alpha = 0.1

#For range of alpha values
alpha_variable = []
for i in range(-5,1):
	alpha_variable.append(2**i)

# split into train and test
num_samples = len(all_data)
all_idx = list(range(num_samples))
random.shuffle(all_idx)
idx_limit = int(0.8*num_samples)
train_idx = all_idx[:idx_limit]
test_idx = all_idx[idx_limit:]
train_examples = [all_data[ii] for ii in train_idx]
test_examples = [all_data[ii] for ii in test_idx]


# Preprocess train and test examples
train_words = []
train_labels = []
test_words = []
test_labels = []

# train examples
for line in train_examples:
    line = line.strip('\r\n\t ')  # remove trailing spaces, tabs and carraige returne
    line = line.lower()  # lowercase
    line = line.replace("\t", ' ')  # convert tabs to spae
    line_words = re.findall(r'\w+', line)
    line_words = [xx for xx in line_words if xx != '']  # remove empty words

    label = line_words[0]
    label = 1 if label == 'spam' else 0
    line_words = line_words[1:]
    train_words.append(line_words)
    train_labels.append(label)
    
# test examples
for line in test_examples:
    line = line.strip('\r\n\t ')  # remove trailing spaces, tabs and carraige return
    line = line.lower()  # lowercase
    line = line.replace("\t", ' ')  # convert tabs to spae
    line_words = re.findall(r'\w+', line)
    line_words = [xx for xx in line_words if xx != '']  # remove empty words

    label = line_words[0]
    label = 1 if label == 'spam' else 0

    line_words = line_words[1:]
    test_words.append(line_words)
    test_labels.append(label)

spam_words = []
ham_words = []
for ii in range(len(train_words)):  # we pass through words in each (train) SMS
    words = train_words[ii]
    label = train_labels[ii]
    if label == 1:
        spam_words += words
    else:
        ham_words += words
input_words = spam_words + ham_words  # all words in the input vocabulary

def count_occurance(alpha):
	# Count spam and ham occurances for each word
	spam_counts = {}; ham_counts = {}
	# Spamcounts
	for word in spam_words:
	    try:
	        word_spam_count = spam_counts.get(word)
	        spam_counts[word] = word_spam_count + 1
	    except:
	        spam_counts[word] = 1 + alpha  # smoothening

	for word in ham_words:
	    try:
	        word_ham_count = ham_counts.get(word)
	        ham_counts[word] = word_ham_count + 1
	    except:
	        ham_counts[word] = 1 + alpha  # smoothening
	return spam_counts,ham_counts

num_spam = len(spam_words)
num_ham = len(ham_words)

#Calculating the Prior (Probability of Spam or Ham)
PHW = 0
PSW = 0
for label in train_labels:
    if(label==1):
        PSW = PSW + 1
    else:
        PHW = PHW + 1
PS = PSW/(PSW+PHW)
PH = PHW/(PSW+PHW)

#Calculating the Posterior (Probability of word being a Spam or Ham)
def predict(test_words,alpha,spam_counts,ham_counts):
	predictions = []
	for i in range(len(test_words)):
		PWH = 1
		PWS  = 1
		ind_message = test_words[i]
		for words in ind_message:
			try:
				PWS *= ((spam_counts[words] + alpha) / (num_spam + (alpha * 20000)))
			except:
				PWS *= (alpha/(num_spam + (alpha * 20000)))
			try:
				PWH *= ((ham_counts[words] + alpha) / (num_ham + (alpha * 20000)))
			except:
				PWH *= (alpha/(num_ham + (alpha * 20000)))
		#print(PWS,PWH)
		PSW = PWS * PS
		PHW = PWH * PH
		if PSW > PHW:
			test_label = 1
		else:
			test_label = 0
		#print(PHW,PSW)
		predictions.append(test_label)
	#print(predictions)
	return predictions

#Calculate the True Positive, True Negative, False Positive and False Negative for Train/Test Data
def calc_confusion_matrix(test_labels, predictions):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    for i in range(len(test_labels)):
        true_positive += int(test_labels[i] == 1 and predictions[i] == 1)
        true_negative += int(test_labels[i] == 0 and predictions[i] == 0)
        false_positive += int(test_labels[i] == 0 and predictions[i] == 1)
        false_negative += int(test_labels[i] == 1 and predictions[i] == 0)
    return true_positive,true_negative,false_positive,false_negative

#Function to print the Confusion Matrix
def print_confusion_matrix(true_positive,true_negative,false_positive,false_negative):
    print('\n' + 'True Positive: ' + str(true_positive) + '\t' + 'False Positive: ' + str(false_positive))
    print('False Negative: ' + str(false_negative) + '\t' + 'True Negative: ' + str(true_negative) + '\n')
    return

#Function to Calculate Accuracy for Train and Test Data
def calc_accuracy(true_positive,true_negative,false_positive,false_negative):
	accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
	return accuracy

#Function to calculate the Precision for Train and Test Data
def calc_precision(true_positive,false_positive):
	precision = true_positive / (true_positive + false_positive)
	return precision

#Function to calculate the Recall for Train and Test Data
def calc_recall(true_positive,false_negative):
	recall = true_positive / (true_positive + false_negative)
	return recall

#Function to calculate the F-Score for Train and Test Data
def calc_f_score(precision,recall):
	f_score = 2 * precision * recall / (precision + recall)
	return f_score

#Function to plot a line plot of Accuracy for Train and Test Data
def line_plot_accuracy(train_variable,test_variable,alpha_variable):
	plt.plot(alpha_variable, train_variable,label = 'Train Accuracy',marker = 'o')
	plt.plot(alpha_variable, test_variable,label = 'Test Accuracy',marker = 'o')

#Function to plot a line plot of F-Score for Train and Test Data
def line_plot_fscore(train_variable,test_variable,alpha_variable):
	plt.plot(alpha_variable, train_variable,label = 'Train F-score',marker = 'o')
	plt.plot(alpha_variable, test_variable,label = 'Test F-score',marker = 'o')

#Function to Display plots in Subplots
def show_plot(alpha_variable,accuracy_variable_train,accuracy_variable_test,f_score_variable_train,f_score_variable_test):
	plt.figure(figsize=(10,9))
	accuracy_plot = plt.subplot(2,1,1)
	line_plot_accuracy(accuracy_variable_train,accuracy_variable_test,[x for x in range(-5,1)])
	plt.legend()
	plt.title('Train Accuracy vs Test Accuracy')
	plt.ylabel('Accuracy')

	f_score_plot = plt.subplot(2,1,2,sharex = accuracy_plot,sharey = accuracy_plot)
	line_plot_fscore(f_score_variable_train,f_score_variable_test,[x for x in range(-5,1)])
	plt.legend()
	plt.title('Train F-Score vs Test F-Score')
	plt.xlabel('alpha')
	plt.ylabel('F-Score')

	plt.show()

#Main Function and Calculations
def main():
	accuracy_variable_test = []
	f_score_variable_test = []
	accuracy_variable_train = []
	f_score_variable_train = []
	spam_counts,ham_counts = count_occurance(alpha)
	predictions = predict(test_words,alpha,spam_counts,ham_counts)
	true_positive,true_negative,false_positive,false_negative = calc_confusion_matrix(test_labels,predictions)
	print('Aplha:',  alpha)
	print('\n')
	print_confusion_matrix(true_positive,true_negative,false_positive,false_negative)
	accuracy = calc_accuracy(true_positive,true_negative,false_positive,false_negative)
	precision = calc_precision(true_positive,false_positive)
	recall = calc_recall(true_positive,false_negative)
	f_score = calc_f_score(precision,recall)
	print("\nAccuracy: ",accuracy)
	print("Precision: ",precision)
	print("Recall: ",recall)
	print("F_Score: ",f_score)

	#For different values of alpha, Test Data
	#print('2**alpha' +'Accuracy' + '\t' + 'Precision' + '\t' + 'Recall' + '\t' + 'F-Score' + '\n')
	for alp in range(len(alpha_variable)):
		spam_counts,ham_counts = count_occurance(alpha_variable[alp])
		predictions = predict(test_words,alpha_variable[alp],spam_counts,ham_counts)
		true_positive,true_negative,false_positive,false_negative = calc_confusion_matrix(test_labels,predictions)
		accuracy = calc_accuracy(true_positive,true_negative,false_positive,false_negative)
		precision = calc_precision(true_positive,false_positive)
		recall = calc_recall(true_positive,false_negative)
		f_score = calc_f_score(precision,recall)
		#print("Accuracy: ",accuracy)
		#print("Precision: ",precision)
		#print("Recall: ",recall)
		#print("F_Score: ",f_score)
		#print(str(alpha_variable[alp]) + '\t' +str(np.round(accuracy,11)) + '\t' + str(np.round(precision,11)) + '\t' + str(np.round(recall,11)) + '\t' + str(np.round(f_score,11)) + '\n')
		#print_confusion_matrix(true_positive,true_negative,false_positive,false_negative)
		accuracy_variable_test.append(accuracy)
		f_score_variable_test.append(f_score)
	
	#For different values of alpha, Train Data
	#print('2**alpha' +'Accuracy' + '\t' + 'Precision' + '\t' + 'Recall' + '\t' + 'F-Score' + '\n')
	for alp in range(len(alpha_variable)):
		spam_counts,ham_counts = count_occurance(alpha_variable[alp])
		predictions = predict(train_words,alpha_variable[alp],spam_counts,ham_counts)
		true_positive,true_negative,false_positive,false_negative = calc_confusion_matrix(train_labels,predictions)
		accuracy = calc_accuracy(true_positive,true_negative,false_positive,false_negative)
		precision = calc_precision(true_positive,false_positive)
		recall = calc_recall(true_positive,false_negative)
		f_score = calc_f_score(precision,recall)
		print('Aplha:',  alpha_variable[alp])
		print('\n')
		#print("Accuracy: ",accuracy)
		#print("Precision: ",precision)
		#print("Recall: ",recall)
		#print("F_Score: ",f_score)
		#print(str(alpha_variable[alp]) + '\t' +str(np.round(accuracy,11)) + '\t' + str(np.round(precision,11)) + '\t' + str(np.round(recall,11)) + '\t' + str(np.round(f_score,11)) + '\n')
		#print_confusion_matrix(true_positive,true_negative,false_positive,false_negative)
		accuracy_variable_train.append(accuracy)
		f_score_variable_train.append(f_score)
		#print(accuracy_variable)
		#print(f_score_variable)
		#print(alpha_variable)
	
	show_plot(alpha_variable,accuracy_variable_train,accuracy_variable_test,f_score_variable_train,f_score_variable_test)
	
if __name__ == '__main__':
	main()
	