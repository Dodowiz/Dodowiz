<h1 align="center">Depression Detection- AmongSocialMedia</h1>
<h3 align="center">Depression is a psychological disorder that affects over three hundred million humans worldwide. A person who is depressed suffers from anxiety in day-to-day life, which affects that person in the relationship with their family and friends, leading to different diseases and in the worst-case death by suicide. With the growth of the social network, most people share their emotion, feelings, their thoughts on social media. If their depression can be detected early by analyzing their post, then by taking the necessary steps, a person can be saved from depression-related diseases, or in the best case he can be saved from committing suicide. In this research work, a hybrid model has been proposed that can detect depression by analyzing users’ textual posts. Deep learning algorithms were trained using the training data and then performance has been evaluated on the test data of the dataset of Reddit which was published for the pilot piece of work. The World Health Organization (WHO) predicts that depression disorders will be widespread in the next 20 years. We try to identify the most effective deep neural network architecture among a few selected architectures that were successfully used in natural language processing tasks. The proposed method employs Recurrent Neural Networks to compute the post representation of each individual. The representations are then combined with other content-based, behavior and living environment features to predict the depression label of the individual with Deep Neural Networks.</h3>

<p>
<img src="https://user-images.githubusercontent.com/97037962/175108641-0e51b16f-d851-42dc-aaaa-72a78220201c.png" width="400" height="400">
  &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;
<img src="https://user-images.githubusercontent.com/97037962/175109125-cb0e746d-eef1-4552-94f5-0126c996a88c.png" width="400" height="400">
</p>

<h2 align="left">WorkFlow </h2>
<p align="left">i)First, we collected the raw dataset from Twitter using Kaggle.</p>
<p align="left">ii)Then we pre-processed the data using ‘nltk’ functions like Tokenizing, POS Tagging, Lemmatizing, Stop Words Removal, lowercase conversion, and duplicate data removal amongst many other methods.</p>
<p align="left">iii)Now, we have sequenced the data using tokenizer() (for CNN-LSTM) and CountVectorizer() (for Machine  Learning). </p>
<p align="left">iv)In the case of the CNN-LSTM model, the text sequences were converted into padded sequences and we sent the training padded sequences and validation padded sequences to the CNN-LSTM model.</p>
<p align="left">v)Finally the models predict the result using the test case and give accuracy as a result. 
</p>

<img src="https://user-images.githubusercontent.com/97037962/175112065-9a48775f-d3e2-4aff-b25c-2618af2a5951.jpeg" width="800" height="400">

<h1>Methodologies of Implementation & Results</h1>
<p>Scraping the data is just the beginning of extracting relevant information from our freshly acquired text data. Data preprocessing is a phase in the data mining and data analysis process that turns raw data into a format that computers and machine learning can understand and evaluate.
Machines want to process data neatly and cleanly, therefore they read input as 1s and 0s. Unstructured data must be cleaned and prepared before being analyzed. We have extracted raw data from Twitter through Kaggle. We have used the following methods to clean our data for this project.
</p>

<h2>A. Data Preprocessing:</h2>
<h3>i)Importing Libraries along with our Data</h3>
<h3>ii)Expanding Contractions</h3>
<h3>iii)Tokenization</h3>
<h3>iv)Converting all Characters to Lowercase</h3>
<h3>v)Removing Punctuations</h3>
<h3>vi)Removing Stopwords</h3>
<h3>vii)Parts of Speech Tagging</h3>
<h3>viii)Lemmatization</h3>

<h2>B. ML classifiers:</h2>
<p>We extracted the feature using countvectorizer()  to prepare our dataset for feeding in ML classifier models. The models we have implemented are given below(with Confusion Matrix as result): </p>
<h3>i)Logistic Regression</h3>

![image](https://user-images.githubusercontent.com/97037962/175115904-762f2294-a875-451e-ab31-efe996ca7a09.png)

<h3>ii)Random Forest</h3>

![image](https://user-images.githubusercontent.com/97037962/175115959-b0d9be41-3385-47e6-a6ff-3cfe55bfc97a.png)

<h3>iii)SVM</h3>

![image](https://user-images.githubusercontent.com/97037962/175115987-9d6746d0-9210-4888-b525-ff4ac4902e8b.png)

<h3>iv)Multimodal Naive Bayes</h3>

![image](https://user-images.githubusercontent.com/97037962/175116016-c310425b-9ecf-4ab8-a429-451b8818a880.png)

<h3>v)KNN</h3>

![image](https://user-images.githubusercontent.com/97037962/175116054-2fd5fe2d-8012-44bd-9915-2d751750270a.png)

<h3>vi)Decision Tree</h3>

![image](https://user-images.githubusercontent.com/97037962/175116089-2a172a4e-23c6-4e9b-8f3e-bd3bf68cdb56.png)
<h2>C.CNN-LSTM Model:</h2>
<h3>CNN:</h3>
<p>
A Convolutional Neural Network (ConvNet/CNN) is a Deep Learning system that can take an input image, assign relevance (learnable weights and biases) to various aspects/objects in the image, and distinguish between them. When compared to other classification methods, the amount of pre-processing required by a ConvNet is significantly less. While filters are hand-engineered in basic approaches, ConvNets can learn these filters/characteristics with enough training.</p>
<h3>LSTM:</h3>
<p>These are an improvement over LSTMs. Each training sequence is presented forward and backward in bidirectional LSTMs to independent recurrent nets. The output layer for both sequences is the same. Bidirectional LSTMs have comprehensive knowledge of every point in a sequence, including everything that comes before and after it.Conventional recurrent neural networks can only collect information from the prior context. Bidirectional LSTMs, on the other hand, gather information by processing data in both directions within two hidden layers, which are pushed toward the same output
layer. In both directions, this allows bidirectional LSTMs to access long-range context.
</p>

<img src="https://user-images.githubusercontent.com/97037962/175117696-ea48c02a-8aa1-4c04-a10f-5959d23ddcb1.png" width="400" height="200">

<h1>Visualisation of Data Prediction by CNN_LSTM model using Scatter plot:</h1>

![image](https://user-images.githubusercontent.com/97037962/175118374-8083dbf7-2bbf-4c2d-8fa7-549dfed6e9ad.png)

<p>Sea green: Depressed</p>
<p>pink:Not Depressed</p>


<h1>Accuracy Comparison among models:</h1>

<img src="https://user-images.githubusercontent.com/97037962/175118412-3453035d-2b93-4e73-bfc7-2a120581404a.png" width="600" height="600">

<h3 align="left">Connect with me:</h3>
<p align="left">
<a href="https://linkedin.com/in/https://www.linkedin.com/in/debrit-bhattacharyya-77622a210/" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/linked-in-alt.svg" alt="https://www.linkedin.com/in/debrit-bhattacharyya-77622a210/" height="30" width="40" /></a>
<a href="https://fb.com/https://www.facebook.com/debrit.bhattacharyya" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/facebook.svg" alt="https://www.facebook.com/debrit.bhattacharyya" height="30" width="40" /></a>
<a href="https://instagram.com/https://www.instagram.com/debritbhattacharyya/" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/instagram.svg" alt="https://www.instagram.com/debritbhattacharyya/" height="30" width="40" /></a>
<a href="https://www.hackerrank.com/dodowiz" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/hackerrank.svg" alt="dodowiz" height="30" width="40" /></a>
</p>



<h3 align="left">Languages and Tools:</h3>
<p align="left"> <a href="https://git-scm.com/" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/git-scm/git-scm-icon.svg" alt="git" width="40" height="40"/> </a> <a href="https://www.mysql.com/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/mysql/mysql-original-wordmark.svg" alt="mysql" width="40" height="40"/> </a> <a href="https://pandas.pydata.org/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/2ae2a900d2f041da66e950e4d48052658d850630/icons/pandas/pandas-original.svg" alt="pandas" width="40" height="40"/> </a> <a href="https://www.python.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/> </a> <a href="https://scikit-learn.org/" target="_blank" rel="noreferrer"> <img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" alt="scikit_learn" width="40" height="40"/> </a> <a href="https://seaborn.pydata.org/" target="_blank" rel="noreferrer"> <img src="https://seaborn.pydata.org/_images/logo-mark-lightbg.svg" alt="seaborn" width="40" height="40"/> </a> <a href="https://www.tensorflow.org" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/tensorflow/tensorflow-icon.svg" alt="tensorflow" width="40" height="40"/> </a> </p>
