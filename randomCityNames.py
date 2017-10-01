# LSTM(Long Short Term Memory) Neural Networks to generate city names
import os
from six import moves
import ssl
import tflearn
from tflearn.data_utils import *

# Step1 - Retrieve data 
path = "US_cities.txt"

if not os.path.isfile(path):
	context = ssl._create_unverified_context()
	# Get data set
	moves.urllib.request.urlretrieve("", path, context=context)

# City name max length
max_length = 30

# Vectorize the text file
X, Y, char_idx = textfile_to_semi_redundant_sequences(path, seq_maxlen=max_length, redun_step=3)

# Create LSTM
lstm_model = tflearn.input_data(shape=[None, max_length, len(char_idx)])
lstm_model = tflearn.lstm(lstm_model, 512, return_seq=True)
lstm_model = tflearn.dropout(lstm_model, 0.5)
lstm_model = tflearn.lstm(lstm_model, 512)
lstm_model = tflearn.dropout(lstm_model, 0.5)
lstm_model = tflearn.fully_connected(lstm_model, len(char_idx), activation='softmax')

# Generate City names
generated = tflearn.SequenceGenerator(lstm_model, dictionary=char_idx, 
							seq_maxlen=max_length, 
							clip_gradients=5.0,
							checkpoint_path='model_us_cities')

# Training
for i in range(40):
	seed = random_sequence_from_textfile(path, max_length)
	generated.fit(X, Y, validation_set=0.1, batch_size=128,
		n_epoch=1, run_id='random cities')
	print("Testing.....")
	print(generated.generate(30, temperature=1.2, seq_seed=seed))
	print("Testing.....")
	print(generated.generate(30, temperature=1.0, seq_seed=seed))
	print("Testing.....")
	print(generated.generate(30, temperature=0.5, seq_seed=seed))
