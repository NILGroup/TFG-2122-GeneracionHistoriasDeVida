
from transformers.optimization import  Adafactor 
from transformers import T5Tokenizer, T5ForConditionalGeneration
import warnings
import torch
import warnings
from IPython.display import HTML, display
import matplotlib.pyplot as plt

class Model:
    
    def __init__(self):
        """
        The function initializes the tokenizer, model, optimizer, and device.
        """
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        self.model = T5ForConditionalGeneration.from_pretrained('t5-base', return_dict=True)
        self.__DEFAULT_PARAMS = {"BATCH_SIZE" : 4, "NUM_OF_EPOCHS" : 2}

        if torch.cuda.is_available():
            self.dev = torch.device("cuda:0") 
            print("Running on the GPU")
        else:
            self.dev = torch.device("cpu")
            print("Running on the CPU")
        
        self.model.to(self.dev)
        self.optimizer = Adafactor(
            self.model.parameters(),
            lr=1e-3,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=0.0,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False
        )

        self.global_loss_per_10_steps = []
        self.last_train_loss_per_10_steps = []
    
    def train(self, train_df, batch_size = None, num_of_epochs = None, display_progress = 0):
        """
        The function takes in a dataframe, batch size and number of epochs as input and trains the
        model.
        
        :param train_df: The dataframe containing the training data
        :param batch_size: The number of samples to use for each gradient update
        :param num_of_epochs: The number of epochs to train for
        :param display_progress: If set to 1, it will display a progress bar for each epoch, defaults to
        0 (optional)
        """

        batch_size, num_of_epochs =  self.__checkTrainParams(batch_size, num_of_epochs)

        num_of_batches = len(train_df) / batch_size

        num_of_batches = int(num_of_batches)

        #Sets the module in training mode
        self.model.train()

        self.last_train_loss_per_10_steps=[]
        for epoch in range(1,num_of_epochs+1):
            print('Running epoch: {}'.format(epoch))
            
            running_loss=0

            if(display_progress): out = display(self.__progress(1, num_of_batches+1), display_id=True)
            
            for i in range(num_of_batches):
                inputbatch=[]
                labelbatch=[]
                new_df=train_df[i*batch_size:i*batch_size+batch_size]

                for indx,row in new_df.iterrows():
                    input = row['input_text'] + self.tokenizer.eos_token
                    labels = row['target_text'] + self.tokenizer.eos_token   
                    inputbatch.append(input)
                    labelbatch.append(labels)

                inputbatch=self.tokenizer.batch_encode_plus(inputbatch,padding=True,max_length=400,return_tensors='pt')["input_ids"]
                labelbatch=self.tokenizer.batch_encode_plus(labelbatch,padding=True,max_length=400,return_tensors="pt") ["input_ids"]
                inputbatch=inputbatch.to(self.dev)
                labelbatch=labelbatch.to(self.dev)

                # clear out the gradients of all Variables 
                self.optimizer.zero_grad()

                # Forward propogation
                outputs = self.model(input_ids=inputbatch, labels=labelbatch)
                loss = outputs.loss
                loss_num=loss.item()
                logits = outputs.logits
                running_loss+=loss_num

                if i%10 ==0:    
                    self.global_loss_per_10_steps.append(loss_num)  
                    self.last_train_loss_per_10_steps.append(loss_num)

                if(display_progress): out.update(self.__progress(loss_num,i, num_of_batches+1))

                # calculating the gradients
                loss.backward()

                #updating the params
                self.optimizer.step()
                
            running_loss=running_loss/int(num_of_batches)
            print('Epoch: {} , Running loss: {}'.format(epoch,running_loss))
            

    def encode(self, prompt):
        """
        It takes a prompt, and returns the input_ids of the prompt
        
        :param prompt: The prompt to be encoded
        :return: The input_ids are being returned.
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        return input_ids

    def generateText(self, encode_text, do_sample = False, num_beams = 5, num_beam_groups=1,no_repeat_ngram_size = 2,min_length = 0, max_length = 500, top_k = 50, top_p = 0.95, temperature = 1.0, penalty = 1.0, num_return_sequences = 1, early_stopping=True):
        """
        The function generates text from a given prompt. The prompt is a string of text that is fed to
        the model. The model then generates text based on the prompt. The generated text is returned as
        a list of strings
        
        :param encode_text: The text to encode
        :param do_sample: If False, greedy decoding is used. Otherwise sampling is used. Defaults to
        False, defaults to False (optional)
        :param num_beams: Number of beams for beam search. 1 means no beam search, defaults to 5
        (optional)
        :param num_beam_groups: Number of groups of beams to use. If 1, then the beam search will be
        serialized. If > 1, then the beam search will be parallelized across the groups, defaults to 1
        (optional)
        :param no_repeat_ngram_size: If set to int > 0, all ngrams of that size can only occur once,
        defaults to 2 (optional)
        :param min_length: The minimum length of the sequence to be generated, defaults to 0 (optional)
        :param max_length: The maximum length of the sequence to be generated, defaults to 500
        (optional)
        :param top_k: The number of highest probability vocabulary tokens to keep for top-k-filtering.
        Between 1 and infinity. Default to 50, defaults to 50 (optional)
        :param top_p: The cumulative probability of parameter highest probability vocabulary tokens to
        keep for top-p-filtering
        :param temperature: The higher the temperature, the crazier the text. Try values starting from 0
        (same as greedy) and going up to something like 2 or 3
        :param penalty: The penalty for repeating n-grams. Larger values discourage repetition, but
        values that are too large may result in no n-grams being repeated
        :param num_return_sequences: The number of sequences to generate, defaults to 1 (optional)
        :param early_stopping: If set to True, will stop the beam search as soon as at least
        num_return_sequences have been generated, defaults to True (optional)
        :return: A list of tensors, each tensor is a sequence of tokens.
        """
        #Sets the module in evaluation mode
        self.model.eval()
        encode_text = encode_text.to(self.dev)

        outputs = self.model.generate(encode_text)

        outputs = self.model.generate(encode_text, 
                    do_sample = do_sample,    #Si false devuelve distintas frases
                    num_beams = num_beams,
                    no_repeat_ngram_size = no_repeat_ngram_size, #If set to int > 0, all ngrams of that size can only occur once.
                    num_beam_groups = num_beam_groups,
                    min_length = min_length,
                    max_length = max_length,
                    top_k = top_k,            #The number of highest probability vocabulary tokens to keep for top-k-filtering.                          
                    top_p = float(top_p),          #If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation.
                    temperature = float(temperature),
                    repetition_penalty = penalty,#1.0 No penalty
                    num_return_sequences = num_return_sequences,
                    early_stopping = early_stopping)

        return outputs

    
        
    def decode(self, encode_text):
        """
        It takes a list of integers as input and returns the corresponding text (as a Python string).
        
        :param encode_text: The text to be encoded
        :return: The result is a string of the decoded text.
        """
        result = self.tokenizer.decode(encode_text)
        return result
        

    def plot_global_loss(self):
        """
        It plots the loss of the global network every 10 steps
        """
        steps = [i*100 for i in range(len(self.global_loss_per_10_steps))]
    
        plt.plot(steps, self.global_loss_per_10_steps)
        plt.title('Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.show()


    def plot_last_loss(self):
        """
        It takes the last 10 losses from the training set and plots them.
        """
        steps = [i*100 for i in range(len(self.last_train_loss_per_10_steps))]
    
        plt.plot(steps, self.last_train_loss_per_10_steps)
        plt.title('Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.show()


    def save_model(self, target_url):
        """
        > The function takes in a model and a target url, and saves the model to the target url
        
        :param target_url: The URL of the target model
        """
        self.model.save_pretrained(target_url)


    def load_model(self, source_url):
        """
        > The function takes a source URL as input, downloads the model from the URL, and loads the model
        into the object
        
        :param source_url: The URL of the model to download
        """
        self.model = self.model.from_pretrained(source_url)
        

    def __progress(loss,value, max=100):
        """
        It takes a loss value and a value and max value and returns a progress bar
        
        :param loss: The loss value of the current batch
        :param value: The current value of the progress bar
        :param max: The maximum value of the progress bar, defaults to 100 (optional)
        :return: A string of HTML code.
        """
        return HTML(""" Batch loss :{loss}
            <progress
                value='{value}'
                max='{max}',
                style='width: 100%'
            >
                {value}
            </progress>
        """.format(loss=loss,value=value, max=max))


    def __checkTrainParams(self, batch_size, num_of_epochs):
        """
        If the batch size or number of epochs are not specified, then use the default values.
        
        :param batch_size: The number of samples per gradient update
        :param num_of_epochs: The number of times the model will be trained on the entire dataset
        :return: The batch size and number of epochs.
        """
        if batch_size is None:
            batch_size = self.__DEFAULT_PARAMS["BATCH_SIZE"]

        if num_of_epochs is None:
            num_of_epochs = self.__DEFAULT_PARAMS["NUM_OF_EPOCHS"]


        return batch_size, num_of_epochs

