import scripts.DataGenerator as DataGenerator
import scripts.Dataset as Dataset

import models.Model as Model
#data_generator = DataGenerator.DataGenerator()

#data_generator.generate()
#triples = data_generator.getTriples()
#data_generator.generateFile()

dataset = Dataset.Dataset()
dataset.WebNLG_parser()


model = Model.Model()
print("Ya")
model.train(dataset.train_set_df.iloc[:100,:])
print(model.text())







