# Naive_Bayes

Description :-
------------
A Python program for classification of a message as Ham or Spam using Naive Bayes prediction method, without using Scikit-Learn tools.

Packages Used :-
--------------
1. re
2. random
3. math
4. Numpy --v 1.6.14
5. Matplotlib --v 3.1.0

How to Run the program :-
----------------------
1. Place the file name SMSSpamCollection in the same folder as the code
2. Run the Python file naive_bayes.py using any of the Python interpreter with Python version > 3 (3.6 or 3.7) [Command: python3 naive_bayes.py]
3. Examine the Confusion Matrix and Statistical Inferences printed in the command line
4. Matplotlib will plot a Subplot. Close the plot after viewing to terminate the program
5. If needed to print any Statistical Inference values for different values of alpha, uncomment the print statements within the code

Format of Text File :-
-------------------
The files contain one message per line. Each line is composed by two columns: one with label (ham or spam) and other with the raw text. Here are some examples:

ham   What you doing?how are you?
Spam   Ok lar... Joking wif u oni...
