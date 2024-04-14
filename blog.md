
# Comparing Longformer and BERT in static and dynamic malware analysis
Arnav Chopra, 5234786, A.Chopra-1@student.tudelft.nl

Casper Dekeling, 5275881, C.R.Dekeling@student.tudelft.nl

Emil Malmsten, 5256941, e.l.malmsten@student.tudelft.nl

## Introduction
As technology develops, so does malware. It is continuously evolving, with small mutations and obfuscations constantly being added to existing malware to make it harder to detect. However, this means that most malware has evolved from one of relatively few families. As seen in [1], where malware is visualised as gray-scale images, all of it within such a family still has much in common. Knowing what family every piece of malware originated from would help us understand how these families evolve to generate new types, which in turn will help us detect it. 

In this project, we will use advanced NLP (Natural Language Processing) models, specifically, the BERT (Bidirectional Encoder Representations from Transformers) [2] model and Longformer [3] model, to predict what family a piece of malware originated from, based on features extracted from the software. Our goal is to establish empirically the differences in performance for this task between the Longformer model and the BERT model to see which model performs best on this task. In this blog post, we will give a brief introduction to the BERT and Longformer models, as well as a description of the procedures we took to train, test, and compare the models.

## Background
### Transformer Models
The Transformer architecture, introduced in [4] has revolutionised the field of NLP. It serves as the foundation for BERT and Longformer, but also models like GPT (Generative Pretrained Transformer). It provides a way to handle sequential data without relying on recurrent or convolutional neural networks. 

The key feature of the Transformer architecture is the self-attention mechanism. This allows each element in the input to interact with every other element. For each element, this will calculate a set of attention scores, indicating how important other parts of the input are when encoding that particular element. This enables the model to capture relationships in the data, independent of their distance within the sequence. It further enhances this by using multi-head attention, which is where multiple attention mechanisms are run in parallel, each focusing on a different part of the input sequence. The model then combines the results from these attention mechanisms to get a better understanding of the input.

The Transformer architecture has had a significant impact on the field of NLP, but has also been adapted to multiple other tasks, such as image processing and reinforcement learning. Its ability to efficiently process sequences and capture long-distance dependencies makes it very viable in translation, text summarisation, question answering and more. 

BERT and Longformer are two very prominent Transformer models because of their ability to understand and generate human language. These models have not only revolutionised how machines understand language, but have also been used to analyse complex data structures, such as code within the field of cybersecurity. 

In the realm of NLP, two models have risen to prominence for their remarkable ability to understand, interpret, and generate human language: BERT and Longformer. These models have not only revolutionized how machines understand human language but have also opened up new avenues for analyzing complex data structures, such as code in cybersecurity applications. Understanding the capabilities and relationship between BERT and Longformer is crucial for appreciating their innovative application in malware analysis.

### BERT
Developed by Google, BERT is considered a significant leap in the ability of machines to understand human language. At its core, BERT uses the transformer architecture to understand the context of each word in a sentence, instead of interpreting each word separately. Its success in NLP tasks such as sentence completion and question answering makes it a foundational model for further NLP advancements.

### Longformer
While BERT has shown exceptional performance in NLP tasks by understanding context, it is limited in its input size. The transformer model it relies on is computationally intensive, making it impractical for processing long input sequences. This is the problem that Longformer solves, a model that is designed to overcome this limitation by extending the Transformer model’s ability to handle longer texts. It achieves this by using a modified attention mechanism that is sparse; it focuses on specific segments of the input while ignoring others, reducing the computational cost. This allows Longformer to keep the contextual understanding of BERT while being more efficient at processing significantly longer input sequences. Since our inputs will be the behavior and features of malware programs, which are usually quite large, we expect Longformer to perform better than BERT in being able to classify the different families.

### Malware Analysis
In the context of malware analysis, Longformer’s improvement over BERT in understanding longer input sequences is especially important, as malware code is often complex and lengthy. By using a contextual understanding of the input, Longformer allows for new ways of analysing malware, discovering patterns, features and relationships that traditional methods might miss. This use of NLP technology to analyse malware shows its usefulness beyond the original purpose of the models.

There are two main ways of analysing malware, static analysis and dynamic analysis. Static analysis is based on all the features of a piece of malware that can be extracted without running it. This includes textual analysis of the code and finding specific sequences like certain print messages or URLs accessed. Additionally, it can analyse linked libraries and other metadata, or find packed/encrypted code. Dynamic analysis requires running the program and analysing its runtime behaviour. This can extract features like what functions are called, tracking the values of parameters and the machine instructions executed by the program. [1]

## Methodology
For both models, we will be running static analysis and dynamic analysis separately. We will then compare the results of both models to see which model performs better. To reduce variance in our accuracy and loss metrics, we will be using cross-validation to test our models. We will be running 5 folds, where each fold will test a completely disjoint 20% of the data. For both the loss and accuracy, we will report on the average, standard deviation and standard error of the mean.

### Task division
The tasks are divided as follows:
- Arnav Chopra
  - Extract embeddings for all models
  - Run static analysis on Bert model
  - Setup Kaggle environment
  - Initial data processing for behavior features
  - Run experiments on reduced data for both datasets and both models
- Casper Dekeling
  - Implement cross-validation
  - Run static analysis on Longformer model
  - Run dynamic analysis on Longformer model
  - Use logistic regression as the classifier for both Longformer models (which later was not used)
  - Write blog post
- Emil Malmsten
  - Set up the training loop
  - Set up the data processing for the static features
  - Made the truncation for both static and behaviour features
  - Setup the code for running on Google Cloud (which later was not used)
  - Write the truncation part of the blog-post

## Truncation
Whereas Longformer is made to support long input samples, namely of 4096 tokens, some of our input samples still exceeded this. For BERT, which can handle samples of 512 tokens, even more samples were too large. Therefore, the data needs to be truncated in a meaningful way, that keeps as much relevant information as possible. 

Our approach to truncating data involved initially conducting truncations that only minimally reduced the informational content of the JSON string across all samples, such as removing unnecessary whitespace and shortening key names. Thereafter, for samples exceeding the permissible token count, truncations with information loss were performed, since you can only compress so much without losing information. We did, however, aim to minimize this loss while achieving the required size. 

It is important to note that our expertise in malware classification is limited, necessitating educated guesses about what data is important for accurate classifications.

### Behaviour truncation (Dynamic analysis)
This dataset comprises 90,806 entries, with an average token length of 1,248. Of these samples, 58,375 entries exceed 512 tokens, and 2,755 surpass 4,096 tokens. The behavior JSON consists of a dictionary containing approximately ten lists that describe various malware behaviors, including "files opened" and "attack techniques." We employed a six-step truncation process for these JSONs, applying the first five steps universally and the sixth step only to entries still exceeding the context size, reducing them to the desired limit.

#### 1. Remove Empty Lists 
A notable characteristic of these JSONs is the presence of empty lists, such as "files deleted" when no files have been deleted. The initial step in our truncation strategy was to eliminate these entries. We inferred that the absence of dictionary entries for certain actions like file deletion effectively communicates that no such action occurred, thus their removal entails no informational loss. This adjustment reduced the average token count to 1,192 and decreased the number of entries above 512 tokens by 200 and those above 4,096 by 100.

#### 2. Concatenate Addresses
In the samples there are often many files within the same directory that were, for example, deleted. This would in the JSON often look something like:
                "Win\\Soft\\Micro\\Alpha\\Document",
                "Win\\Soft\\Micro\\Alpha\\Photo",
                "Win\\Soft\\Micro\\Alpha\\Video",
                "Win\\Soft\Micro\\Beta\\Music",
                "Win\\Soft\\Micro\\Beta\\Download",
What this step does is basically make all directories into dictonaries and put the files in recursively, in the above example that would lead to:
json
Win: {
    Soft: {
         Micro: {
              Alpha: {
                    f (files):  [Document, Photo, Video]
                   }
              Beta: {
                   f: [Music, Download]
 }}}}
 
 
This step is the most effective at reducing the average, from 1192 to 828, almost 30% compared to the original average of 1248. It also makes sure that all but 230 entries exceed 4096 tokens.
 
#### 3. Remove Long Sequences
The JSON occasionally contains excessively long sequences of characters, numbers, or alphanumeric combinations within file names, such as:
- 3006406624-3672060426-1000
- AC0A92D435E5
- WCChromeNativeMessagingHost

New sequences within these names are typically separated by dashes, underscores, or spaces. The truncation of these elements varies depending on their type and the desired context size. We generally preserve strings at a relatively long length, reduce numbers more significantly, and drastically shorten alphanumeric combinations. This approach is based on the observation that random alphanumeric combinations generate a high token-per-character ratio, and we believe they contribute minimally to predictive accuracy. Similarly, we assess that lengthy numeric sequences are unlikely to significantly impact predictions. The intent behind maintaining some longer strings is to account for any unusually large, nonsensical strings that might occur.

This truncation strategy significantly reduced the average token count to 622, and only seven entries remained above 4,096 tokens. Additionally, over 10,000 entries were reduced to below 512 tokens.

#### 4. Categorize by Extension
Not only for directories do we have a lot of repetition, but for extensions as well. This step categorizies all f-lists (the innermost files list) into dictonaries as well depending on file type. An example would be:
a.txt
b.txt
d.dll
e
which would be turned into. 
json
txt: [a,b]
dll: [d]
misc: [e]
 
 This step seems to increase the average back up to 648 but we believe that the step is still a good one to have since it removes a lot of token waste on long lists of entries that all have the same extension. 

#### 5. Concatenate Duplicates
Sometimes there are files that all are extremely similar in name, like:
SESSION0001
SESSION0002
SESSION0003
SESSION0004
We see which files share more than 50% similarity (regarding position and letter) and turn them into a number of occurence plus their shared part. In the above example it would become:
4x SESSION000

Considering that this step is based on similarity, the previous step is fairly important for retaining information. Without it, way more useful information would be removed.

After this step, only 3 entries above 4,096 remain. There are 46,460 entries left above 512, 11,900 (20%) less than originally. The average token length is a bit less than half of the original with 599.

#### 6. Iterative Truncation Strategy
We believe that the truncation methods described previously will not significantly impact the integrity of the data, effectively preserving essential information in a more concise format. However, numerous samples will still exceed the desired context length, particularly when the threshold is set at 512 tokens. To address this, we aim to remove as much text as possible from each sample while minimizing information loss. Our strategy involves identifying and eliminating deeply nested single files within the JSON directory structure. This process begins with the smallest lists, typically those containing a single file, and escalates to larger lists if necessary. An example of a complex directory chain that might be targeted for removal is as follows:
json
Win: {
    Soft: {
         Micro: {
              Edgar: {
                    Allan: 
                          Poe:
                             f:  [abc]
}}}}
 
 We believe that these files do not contain too much information while using a heavy amount of tokens.

| Operation             | Average | Over 512 | Over 4096 |
|-----------------------|---------|----------|-----------|
| Original              | 1249    | 58375    | 2755      |
| Remove empty list     | 1192    | 57181    | 2645      |
| Concat addresses      | 828     | 53207    | 230       |
| Remove long sequences | 622     | 47650    | 7         |
| Categorize by extension | 648   | 48422    | 9         |
| Concat duplicates     | 599     | 46460    | 3         |


### Static Truncation
The dataset contains 103,883 samples with an average token length of 1,667. All samples are longer than 512 and 13,212 are longer than 4,096. 

The static dataset is a bit less organized than the behvaviour one.  It is made of several recursive dictonaries and lists and all the samples differ a bit in what information they have. The truncation for the static data is done in 5 steps, where the first four are done for all data and the last step just for the samples still above the context length.

#### 1. Rounding numbers
The first step is simply to round all the floats to 3 significant digits.  This step takes the average down slightly, by 70 tokens. However, it brings down the amount of entries with token above 4,096 to just 744.

#### 2. Shorten keys
The JSON has several long keys to the dictonary, like CNT_INITIALIZED_DATA. We belive that its just as readable for the classifier to have these shorten to 3 letters per word, since most keys still remain distinct. The above example would in this case become CNT_INI_DAT. 

This is pretty significant as it takes the average down by 272.

#### 3. Shorten sequences
The next step is simply to turn all string sequences into shorter versions of themselves. There are sometimes some long sequences of strings. We simply truncate these to length 30 if we have a max token length of 4,096 and to 15 for 512.  

This step decreases the average by 161. It also makes it such that there are only 2 entries above 4,096 (down from 471 after the previous step).

#### 4. Remove Repetitive Key-Value Pairs
In some cases, a high proportion of samples share the same value for certain keys, rendering these data points redundant. For instance, the 'ExitPoint' key is consistently an empty list across all samples. To optimize data size, for a context length of 4,096, we remove key-value (KV) pairs where 90% of the samples share the same value, and for a context length of 512, we remove those where more than 70% are similar. Additionally, 'ordered Opcodes' are always removed since each opcode occurrence is already listed separately.

This step significantly reduces the dataset's size, almost halving the average length and reducing the number of entries exceeding 512 tokens to 20,000 — just one-fifth of the original count. Through these combined truncations, the average token count has been reduced to 40% of its initial value.

#### 5. Iterative Truncation Strategy

This method involves randomly removing data until the target context size is achieved. It calculates the percentage of data still needing reduction and then selectively removes a proportional amount from sub-dictionaries and lists longer than three entries. Given the uncertainty about which data points are most critical, a random selection strategy is used.

It is also important to note that in both the truncation strategy for behaviour and static data, we eliminate all whitespaces between JSON elements, as they are sufficiently delineated by commas, brackets, and colons. Additionally, we remove quotation marks indicating string values to further condense the data format.

| Operation         | Average | Over 512 | Over 4096 |
|-------------------|---------|----------|-----------|
| Original          | 1667    | 103883   | 13212     |
| Round Numbers     | 1601    | 103883   | 744       |
| Shorten Keys      | 1329    | 101576   | 471       |
| Truncate Strings  | 1168    | 99151    | 2         |
| Remove KV pairs   | 660     | 20295    | 2         |


## Implementation
Our implementation will be hosted on Kaggle, an online platform with GPU’s available to run our experiments. Kaggle can host our Python notebooks remotely, meaning we can leave our experiments running without worrying, and our hardware is consistent across experiments. 

### Challenges encountered
The biggest challenge we encountered was Kaggle’s hardware limitations in combination with our large dataset. Each user gets 30 hours of GPU time per week, and each experiment is stopped after running for 12 hours. 
For the BERT model, training and testing the model using cross-validation took 17 hours each, for both the static and dynamic analysis. This meant the experiments had to be spread out over 2 people to be completed in one week, and each person had to split the cross-validation into two experiments. We use seeded randomisation to ensure that the test set of each fold was completely disjoint from the others. 
For the Longformer model, training the model was not feasible within Kaggle’s hardware capabilities. Each input sample for Longformer is 8 times larger than for BERT, and even though Longformer is designed for handling larger input efficiently, it still takes substantially longer to run each fold. To solve this, we attempted to run a modified experiment, where we ran only a forward pass on the data without training the model, and instead of using the fully connected linear layer to classify the data, we used the outputs of the hidden layers and use a logistic regression classifier to classify the data. However, this also required too much resources. As our final solution, we went back to the original plan, of training and testing the model with cross-validation, but with a substantially smaller data size. We limited the data set to 1000 randomly generated samples, instead of the full data size, which had 103883 samples for the static analysis, and 90806 for the dynamic analysis. To give a fair comparison, we also ran BERT with the same 1000 samples. We will however still show the results of the experiments with the full datasets for BERT.

## Results
#### Static analysis on full dataset with BERT:

| Fold                | Accuracy | Loss  |
|---------------------|----------|-------|
| Fold 1              | 0.350    | 5.998 |
| Fold 2              | 0.324    | 5.958 |
| Fold 3              | 0.323    | 5.936 |
| Fold 4 (best)       | 0.398    | 5.956 |
| Fold 5              | 0.341    | 5.999 |
|                     |          |       |
| Average             | 0.347    | 5.970 |
| Standard deviation  | 0.028    | 0.025 |
| Standard mean error | 0.013    | 0.011 |

#### Dynamic analysis on full dataset with BERT:

| Fold                | Accuracy | Loss  |
|---------------------|----------|-------|
| Fold 1              | 0.405    | 5.514 |
| Fold 2              | 0.386    | 5.548 |
| Fold 3              | 0.371    | 5.549 |
| Fold 4 (best)       | 0.423    | 5.489 |
| Fold 5              | 0.403    | 5.503 |
|                     |          |       |
| Average             | 0.397    | 5.521 |
| Standard deviation  | 0.020    | 0.027 |
| Standard mean error | 0.009    | 0.012 |

#### Static analysis on 1000 samples with both models:

| Fold                | Accuracy (BERT) | Accuracy (LF) | Loss (BERT) | Loss (LF) |
|---------------------|-----------------|---------------|-------------|-----------|
| Fold 1              | 0.175           | 0.3           | 7.677       | 7.015     |
| Fold 2              | 0.190           | 0.285         | 7.153       | 6.312     |
| Fold 3              | 0.175           | 0.25          | 7.116       | 6.723     |
| Fold 4              | 0.195           | 0.28          | 7.457       | 6.792     |
| Fold 5              | 0.220           | 0.36          | 7.338       | 6.405     |
| Best fold           | 0.220           | 0.36          | 7.339       | 6.405     |
|                     |                 |               |             |           |
| Average             | 0.191           | 0.295         | 7.348       | 6.649     |
| Standard deviation  | 0.017           | 0.036         | 0.206       | 0.258     |
| Standard mean error | 0.008           | 0.016         | 0.092       | 0.115     |

TODO SAY SOMETHING ABOUT RESULTS AFTER WE GET THE CORRECT RESULTS

#### Dynamic analysis on 1000 samples with both models:

| Fold                | Accuracy (BERT) | Accuracy (LF) | Loss (BERT) | Loss (LF) |
|---------------------|-----------------|---------------|-------------|-----------|
| Fold 1              | 0.250           | 0.285         | 6.876       | 6.598     |
| Fold 2              | 0.245           | 0.245         | 7.695       | 7.486     |
| Fold 3              | 0.285           | 0.300         | 6.980       | 6.753     |
| Fold 4              | 0.260           | 0.290         | 7.544       | 7.340     |
| Fold 5              | 0.265           | 0.260         | 6.707       | 6.574     |
| Best fold           | 0.285           | 0.300         | 6.980       | 6.753     |
|                     |                 |               |             |           |
| Average             | 0.261           | 0.276         | 7.160       | 6.950     |
| Standard deviation  | 0.014           | 0.020         | 0.388       | 0.386     |
| Standard mean error | 0.006           | 0.009         | 0.174       | 0.173     |

TODO SAY SOMETHING ABOUT RESULTS AFTER WE GET THE CORRECT RESULTS

## Conclusion/Future work
TODO AFTER ACTUAL RESULTS

TODO SAY SOMETHING ABOUT HOW USEFUL NLP IS FOR MALWARE ANALYSIS

### Future work
Further experiments can be run using the Longformer model which use more resources, to run the full experiment as it was intended, training and testing the model with cross-validation, and using the intended classification layer instead of the logistic regression model.
Additionally, more research can be done into different algorithms to truncate the data, empirically comparing different methods.


## References
[1] Gibert, D., Mateu, C., Planes, J. et al. Using convolutional neural networks for classification of malware represented as images. J Comput Virol Hack Tech 15, 15–28 (2019). https://doi.org/10.1007/s11416-018-0323-0

[2] Devlin, J., Chang, M., Lee, K., Toutanova, K. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (2018).  https://doi.org/10.48550/arXiv.1810.04805

[3] Beltagy, I., Peters, M., Cohan, A. Longformer: The Long-Document Transformer (2020). 
https://doi.org/10.48550/arXiv.2004.05150

[4] Vaswani, A., Shazeer, N., Parmar, N., et al. Attention Is All You Need (2017). 
https://doi.org/10.48550/arXiv.1706.03762


