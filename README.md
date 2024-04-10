# Seq2seq
This is the python class provide seq2seq model both learning and predicting. And based Keras.

## Example of workflow
1. Create a instance with some parameter.  
2. Convert the data you want to train or infer into a subsequence.  
3. Train the model.  
4. Predict test data.  

## Function explain
### Seq2Seq.\_\_init\_\_()
Initiate some class variables and define models both learning and predicting.<br>
--args--  
・window : Window size of subsequence  
・input_dim : Dimention size of input data  
・latent_dim : hidden layer size  
・epochs : epoch size  
・batch_size : batch size  

### Seq2Seq.create_learning_model()
This function define learning model like below images.<br>
![seq2seq_training_process_white_bg](https://github.com/xerkey/seq2seq/assets/87689622/3d82b28c-1209-4853-88f0-737d45e62042)<br>
--args--  
Nothing

### Seq2Seq.create_pred_model()
This function define predicting model like below images.<br>
![seq2seq_prediction_process_white_bg](https://github.com/xerkey/seq2seq/assets/87689622/5c2e6178-b823-428e-9ed6-889ba8a1bef2)<br>
--args--  
Nothing

### Seq2Seq.create_subseq()
This function create subsequence.  <br>
--args--  
・data : 1 dimentional list object.  
・stride : size of shifting.  
・window : window size.  

### Seq2Seq.learn()
Excution learning. create subsequence automatically.<br>
--args--  
・input : input data.
・stride : size of shifing.

### Seq2Seq.pred()
Excution prediction. create subsequence automatically.<br>
--args--  
・input : input data.
