# CS5043_Assignment6

## Data and provided code
###Goal: 
Our data consists of sequences of amino acids, each making a protein I think. We'd like to be able to predict what 
"family" our amino acids belong to, with four families present in this dataset (PF01925, PF01810, PF02659 or PF03824). 

###Data: 
The data is partitioned into five independent folds, with the four classes distributed equally across the five folds. 
However, they each have a different number of examples, with as much as a 1-10 ratio between the minority and majority classes.

Each example consists of the following:
- a tokenized string of length 1340 amino acids. The strings in the data set have been padded this
time on the left hand side, wheras before the padding was on the right. Additionally, there is a token that corresponds to an 
"unknown" token.
- A tokenized class label (integers from 0 to 4)

Two separate ways to load the data are provided. 
- prepare_data_set(), which loads the raw data, constructs the train/validation/test data sets, and performs the tokenization. These files are smaller, but require CPU processing before training (so could take longer to run?)
- load_rotation(): loads an already constructed rotation from a pickle file. These files are a lot larger, but require no processing once loaded. 

## Deep Learning Experiments

### Objective: 
Create a nn that can predict the family of a given amino acid. I need to compare a "simple" model with a "complex" one. Unsure what those will consist of, definitions are up to me. 
Notes:
- Consider a combination of 1D CNNs and RNNs. The CNNs give an opportunity to collapse info from multiple steps into a single step, reducing the length for the subsequent RNN layer. 
- Network should have four outputs, one for each class. Use softmax for the final step.
- Class labels from the loader are integers, not one-hot encoded (why?). I can convert integers to 1-hot encoded and use **categorical cross-entropy** for loss, or keep the integers and use **sparse categorical cross-entropy**, which will do the conversion for me
- If using the sparse CE, I'll need to use sparse cat accuracy as well. 


## Journal

### 04/10/22
Goal for today: Just get a working file up and running. If I can do that then everything after today will be just an improvement. 
