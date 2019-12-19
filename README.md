# Tensorflow Project Template
## This work is a modification version based on the [original template](https://github.com/MrGemy95/Tensorflow-Project-Template) from [MrGemy95](https://github.com/MrGemy95)   

(This template is tested with Python 3.7 and TensorFlow 1.x)

# Very Quick Start
- Training: python mnist_main.py -c ./config/mnist_config.json
- Evaluation: python model_evaluation.py (you need to modify the file directly for input)

# Table Of Contents

-  [In a Nutshell](#in-a-nutshell)
-  [In Details](#in-details)
    -  [Project architecture](#project-architecture)
    -  [Folder structure](#folder-structure)
    -  [ Main Components](#main-components)
        -  [Models](#models)
        -  [Trainer](#trainer)
        -  [Data Loader](#data-loader)
        -  [Logger](#logger)
        -  [Configuration](#configuration)
        -  [Main](#main)
 -  [Future Work](#future-work)
 -  [Contributing](#contributing)
 -  [Acknowledgments](#acknowledgments)

# In a Nutshell   
In a nutshell here's how to use this template, so **for example** assume you want to implement VGG model so you should do the following:
-  In models folder create a class named VGG that inherit the "base_model" class

```python

    class MNISTModel(BaseModel):
        def __init__(self, config):
            super(VGGModel, self).__init__(config)
            #call the build_model and init_saver functions.
            self.build_model() 
            self.init_saver() 
  ```
- Override these two functions "build_model" where you implement the vgg model, and "init_saver" where you define a tensorflow saver, then call them in the initalizer.
    
```python
     def build_model(self):
        # here you build the tensorflow graph of any model you want and also define the loss.
        pass
            
     def init_saver(self):
        # here you initalize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

  ```
   
- In trainers folder create a MNIST trainer that inherit from "base_train" class
```python

    class MNISTTrainer(BaseTrain):
        def __init__(self, sess, model, data, config, logger):
            super(VGGTrainer, self).__init__(sess, model, data, config, logger)
```
- Override these two functions "train_step", "train_epoch" where you write the logic of the training process
```python

    def train_epoch(self):
        """
       implement the logic of epoch:
       -loop on the number of iterations in the config and call the train step
       -add any summaries you want using the summary
        """
        pass

    def train_step(self):
        """
       implement the logic of the train step
       - run the tensorflow session
       - return any metrics you need to summarize
       """
        pass

```
- In main file, you create the session and instances of the following objects "Model", "Logger", "Data_Generator", "Trainer", and config
```python
    sess = tf.Session()
    # create instance of the model you want
    model = VGGModel(config)
    # create your data generator
    data = DataGenerator(config)
    # create tensorboard logger
    logger = Logger(sess, config)
```
- Pass the all these objects to the trainer object, and start your training by calling "trainer.train()" 
```python
    trainer = VGGTrainer(sess, model, data, config, logger)

    # here you train your model
    trainer.train()

```


# In Details


Folder structure
--------------

```
├──  base
│   ├── base_model.py   - this file contains the abstract class of the model.
│   └── base_train.py   - this file contains the abstract class of the trainer.
│
│
├── model               - this folder contains any model of your project.
│   └── mnist_model.py
├── config               - this folder contains training config settings in json format.
│   └── mnist_model.py
├── trainer             - this folder contains trainers of your project.
│   └── mnist_trainer.py
│   
├──  wrappers  
│    └── layer_wrappers.py  - here's the wrappers for frequent used layers in CNN model.  
├──  data_generator  
│    └── data_generator.py  - here's the data_generator that is responsible for all data handling.
│ 
└── utils
     ├── logger.py  - here's text logging, a wrapper of Python's logging.
     ├── summary.py - here's TensorFlow summary setting.
     └── utils.py - here's other utilities.


```
  

## Main Components

### Models
--------------
- #### **Base model**
    
    Base model is an abstract class that must be Inherited by any model you create, the idea behind this is that there's much shared stuff between all models.
    The base model contains:
    - ***Save_checkpoint*** -This function to save a checkpoint to the desk.
  - - ***Save_protobuf*** -This function to save a checkpoint to the desk. 
    - ***Load*** -This function to load a checkpoint from the desk.
    - ***current_epoch, current_iter, Global_step counters*** -These variables to keep track of the current epoch and global step.
    - ***Init_Saver*** An abstract function to initialize the saver used for saving and loading the checkpoint, ***Note***: override this function in the model you want to implement.
    - ***Build_model*** Here's an abstract function to define the model, ***Note***: override this function in the model you want to implement.
- #### **Your model**
    Here's where you implement your model.
    So you should :
    - Create your model class and inherit the base_model class
    - override "build_model" where you write the tensorflow model you want
    - override "init_save" where you create a tensorflow saver to use it to save and load checkpoint
    - call the "build_model" and "init_saver" in the initializer.

### Trainer
--------------
- #### **Base trainer**
    Base trainer is an abstract class that just wrap the training process.
    
- #### **Your trainer**
     Here's what you should implement in your trainer.
    1. Create your trainer class and inherit the base_trainer class.
    2. override these two functions "train_step", "train_epoch" where you implement the training process of each step and each epoch.
### Data Loader
This class is responsible for all data handling and processing and provide an easy interface that can be used by the trainer.
### Logger
This class is responsible for the tensorboard summary, in your trainer create a dictionary of all tensorflow variables you want to summarize then pass this dictionary to logger.summarize().


This class also supports reporting to **Comet.ml** which allows you to see all your hyper-params, metrics, graphs, dependencies and more including real-time metric.
Add your API key [in the configuration file](configs/example.json#L9):

For example: "comet_api_key": "your key here"





### Configuration
I use Json as configuration method and then parse it, so write all configs you want then parse it using "utils/config/process_config" and pass this configuration object to all other objects.
### Main
Here's where you combine all previous part.
1. Parse the config file.
2. Create a tensorflow session.
2. Create an instance of "Model", "Data_Generator" and "Logger" and parse the config to all of them.
3. Create an instance of "Trainer" and pass all previous objects to it.
4. Now you can train your model by calling "Trainer.train()"

