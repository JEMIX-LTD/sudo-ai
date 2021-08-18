# Trainer

## Module Content

Trainer module
@author: Aymen Jemi (jemix) <[jemiaymen@gmail.com](mailto:jemiaymen@gmail.com)> at SUDO-AI

sudoai.trainer is a lightweight and extensible library to
easily train model for Natural Language Processing (NLP).

**WARNING**: To try trainer dataset must be exist in
drive or local storage.

### Examples

These examples illustrate how to use training modules.

Word to Label classification:

```python
>>> trainer = TokenClassificationTrainer(
>>>     id='w2l_dataset_model',
>>>     hidden_size=12,
>>>     lr=0.002,
>>>     momentum=0.2,
>>>     loss='nll',
>>>     optimizer='rmsprop',
>>>     do_eval=True,
>>>     do_shuffle=True,
>>>     epochs=1)
>>> trainer(hyperparam=True, test_mode=True)
```


### class sudoai.trainer.HybridXMLTrainer(id: str, model: sudoai.models.xmltc.HybridXMLTC = None, version: str = '0.1.0', teacher_forcing_ratio: float = 0.5, hidden_size: int = 512, print_every: int = 30000, lr: float = 0.0001, epochs: int = 2, drop_out: float = 0.1, do_eval: bool = False, do_save: bool = False, do_shuffle: bool = True, split_ratio: float = 0.2, do_save_checkpoint: bool = False, continue_from_checkpoint: str = None, loss: str = 'nll', optimizer: str = 'rmsprop', momentum: float = 0.0, wandb: <module 'wandb' from 'c:\\\\anaconda3\\\\lib\\\\site-packages\\\\wandb\\\\__init__.py'> = None, wandb_key: str = None, multiclass: bool = False, base_path: str = None)
Bases: `sudoai.trainer.core.Trainer`

HybridXML Trainer class.

Trainer for HybridXML model.


#### id()
Dataset identifier.


* **Type**

    str



#### model()
HybridXML Model. Defaults to None.


* **Type**

    `HybridXML`, optional



#### version()
Model version. Defaults to ‘0.1.0’.


* **Type**

    str, optional



#### teacher_forcing_ratio()
Teacher Forcing ratio for acceleration. Defaults to 0.5.


* **Type**

    float, optional



#### hidden_size()
Hidden size of the model. Defaults to 512.


* **Type**

    int, optional



#### print_every()
Log frequency result by steps (log result every 30000 steps). Defaults to 30000.


* **Type**

    int, optional



#### lr()
Learn rate value. Defaults to 0.0001.


* **Type**

    float, optional



#### epochs()
Number of epochs. Defaults to 2.


* **Type**

    int, optional



#### drop_out()
Drop out value. Defaults to 0.1.


* **Type**

    float, optional



#### do_eval()
If True evaluate. Defaults to False.


* **Type**

    bool, optional



#### do_save()
If True save the model when training ends. Defaults to False.


* **Type**

    bool, optional



#### do_shuffle()
If True shuffle the dataset. Defaults to True.


* **Type**

    bool, optional



#### split_ratio()
Split ratio (0.2 meaning 80% training and 20% validation). Defaults to 0.2.


* **Type**

    float, optional



#### do_save_checkpoint()
If True save checkpoint every epoch. Defaults to False.


* **Type**

    bool, optional



#### continue_from_checkpoint()
Path of checkpoint as start point. Defaults to None.


* **Type**

    str, optional



#### loss()
Loss function. Defaults to ‘nll’.


* **Type**

    str, optional



#### optimizer()
Optimizer function. Defaults to ‘rmsprop’.


* **Type**

    str, optional



#### momentum()
Momentum value for optimizer. Defaults to 0.0.


* **Type**

    float, optional



#### wandb()
Wandb class to track logs. Defaults to None.


* **Type**

    wandb, optional



#### wandb_key()
Wandb api key. Defaults to None.


* **Type**

    str, optional



#### base_path()
Base path to save checkpoint and model. Defaults to None.


* **Type**

    str, optional



#### multiclass()
If True multiclass. Defaults to False.


* **Type**

    bool, optional



#### data_process(test_mode: bool = False)
Process dataset.

Split data for training and evaluation.
:param test_mode: If True enter in test mode. Defaults to False.
:type test_mode: bool, optional


* **Raises**

    **Exception** – when dataset is too small.



* **Returns**

    Dict with data for training and evaluation.



* **Return type**

    dict



#### evaluate(n=10)
Visual evaluation process for random input from dataset


* **Parameters**

    **n** (*int**, **optional*) – number of item to evaluate. Defaults to 10.



#### save()
Save model with id and version.


* **Returns**

    Path of model saved to.



* **Return type**

    str



#### save_checkpoint(num_epoch: int, train_loss_avg: float, eval_loss_avg: Optional[float] = None)
Save checkpoint


* **Parameters**

    
    * **num_epoch** (*int*) – Current epoch number.


    * **train_loss_avg** (*float*) – Current average training loss .


    * **eval_loss_avg** (*float**, **optional*) – Current average evaluation loss. Defaults to None.



#### start(hyperparam: bool = False, test_mode: bool = False, log_history: bool = False)
Entry point to begin the training process.


* **Parameters**

    
    * **hyperparam** (*bool**, **optional*) – If True process in HyperParameter mode. Defaults to False.


    * **test_mode** (*bool**, **optional*) – If True process in test mode


    * **mode to try with small amount of data****)**** Defaults to False.** (*(**Test*) – 


    * **log_history** (*bool**, **optional*) – If True logs all steps for reporting. Defaults to False.


**WARNING**: If your dataset is too small (less then 100) and you chose test mode,
an Exception raised.


* **Raises**

    **ModelError** – when model is None.



* **Returns**

    accuracy and loss.
    list: Log History.



* **Return type**

    dict



#### steps(train, num_epoch=1, hyperparam=False, log_history=False)
Training steps


* **Parameters**

    
    * **train** (*dict*) – Dict with data for training and evaluation.


    * **num_epoch** (*int**, **optional*) – Current epoch number. Defaults to 1.


    * **hyperparam** (*bool**, **optional*) – If True enter Hypertuning mode. Defaults to False.


    * **log_history** (*bool**, **optional*) – If True log history. Defaults to False.



* **Returns**

    accuracy and loss.
    list: Log History.



* **Return type**

    dict



### exception sudoai.trainer.ModelError(model, message='Model error.')
Bases: `Exception`

Exception raised in Trainer class.


* **Parameters**

    
    * **model** (`torch.nn.Module`) – Current Model object.


    * **message** (*str**, **optional*) – Human readable string describing the exception. Defaults to “Model error.”.



#### model()
Current Model object.


* **Type**

    `torch.nn.Module`



#### message()
Human readable string describing the exception. Defaults to “Model error.”.


* **Type**

    str



#### args()

#### with_traceback()
Exception.with_traceback(tb) –
set self.__traceback__ to tb and return self.


### class sudoai.trainer.Seq2LabelTrainer(id: str, model: sudoai.models.seq.Seq2Label = None, version: str = '0.1.0', teacher_forcing_ratio: float = 0.5, hidden_size: int = 512, print_every: int = 30000, lr: float = 0.0001, epochs: int = 2, drop_out: float = 0.1, do_eval: bool = False, do_save: bool = False, do_shuffle: bool = True, split_ratio: float = 0.2, do_save_checkpoint: bool = False, continue_from_checkpoint: str = None, loss: str = 'nll', optimizer: str = 'rmsprop', momentum: float = 0.0, wandb: <module 'wandb' from 'c:\\\\anaconda3\\\\lib\\\\site-packages\\\\wandb\\\\__init__.py'> = None, wandb_key: str = None, base_path: str = None)
Bases: `sudoai.trainer.core.Trainer`

Sequence to Label Trainer class.

Trainer for sequence to label model.


#### id()
Dataset identifier.


* **Type**

    str



#### model()
Sequence To Label Model. Defaults to None.


* **Type**

    `Seq2Label`, optional



#### version()
Model version. Defaults to ‘0.1.0’.


* **Type**

    str, optional



#### teacher_forcing_ratio()
Teacher Forcing ratio for acceleration. Defaults to 0.5.


* **Type**

    float, optional



#### hidden_size()
Hidden size of the model. Defaults to 512.


* **Type**

    int, optional



#### print_every()
Log frequency result by steps (log result every 30000 steps). Defaults to 30000.


* **Type**

    int, optional



#### lr()
Learn rate value. Defaults to 0.0001.


* **Type**

    float, optional



#### epochs()
Number of epochs. Defaults to 2.


* **Type**

    int, optional



#### drop_out()
Drop out value. Defaults to 0.1.


* **Type**

    float, optional



#### do_eval()
If True evaluate. Defaults to False.


* **Type**

    bool, optional



#### do_save()
If True save the model when training ends. Defaults to False.


* **Type**

    bool, optional



#### do_shuffle()
If True shuffle the dataset. Defaults to True.


* **Type**

    bool, optional



#### split_ratio()
Split ratio (0.2 meaning 80% training and 20% validation). Defaults to 0.2.


* **Type**

    float, optional



#### do_save_checkpoint()
If True save checkpoint every epoch. Defaults to False.


* **Type**

    bool, optional



#### continue_from_checkpoint()
Path of checkpoint as start point. Defaults to None.


* **Type**

    str, optional



#### loss()
Loss function. Defaults to ‘nll’.


* **Type**

    str, optional



#### optimizer()
Optimizer function. Defaults to ‘rmsprop’.


* **Type**

    str, optional



#### momentum()
Momentum value for optimizer. Defaults to 0.0.


* **Type**

    float, optional



#### wandb()
Wandb class to track logs. Defaults to None.


* **Type**

    wandb, optional



#### wandb_key()
Wandb api key. Defaults to None.


* **Type**

    str, optional



#### base_path()
Base path to save checkpoint and model. Defaults to None.


* **Type**

    str, optional



#### data_process(test_mode: bool = False)
Process dataset.

Split data for training and evaluation.
:param test_mode: If True enter in test mode. Defaults to False.
:type test_mode: bool, optional


* **Raises**

    **Exception** – when dataset is too small.



* **Returns**

    Dict with data for training and evaluation.



* **Return type**

    dict



#### evaluate(n=10)
Visual evaluation process for random input from dataset


* **Parameters**

    **n** (*int**, **optional*) – number of item to evaluate. Defaults to 10.



#### save()
Save model with id and version.


* **Returns**

    Path of model saved to.



* **Return type**

    str



#### save_checkpoint(num_epoch: int, train_loss_avg: float, eval_loss_avg: Optional[float] = None)
Save checkpoint


* **Parameters**

    
    * **num_epoch** (*int*) – Current epoch number.


    * **train_loss_avg** (*float*) – Current average training loss .


    * **eval_loss_avg** (*float**, **optional*) – Current average evaluation loss. Defaults to None.



#### start(hyperparam: bool = False, test_mode: bool = False, log_history: bool = False)
Entry point to begin the training process.


* **Parameters**

    
    * **hyperparam** (*bool**, **optional*) – If True process in HyperParameter mode. Defaults to False.


    * **test_mode** (*bool**, **optional*) – If True process in test mode


    * **mode to try with small amount of data****)**** Defaults to False.** (*(**Test*) – 


    * **log_history** (*bool**, **optional*) – If True logs all steps for reporting. Defaults to False.


**WARNING**: If your dataset is too small (less then 100) and you chose test mode,
an Exception raised.


* **Raises**

    **ModelError** – when model is None.



* **Returns**

    accuracy and loss.
    list: Log History.



* **Return type**

    dict



#### steps(train: dict, num_epoch: int = 1, hyperparam: bool = False, log_history: bool = False)
Training steps


* **Parameters**

    
    * **train** (*dict*) – Dict with data for training and evaluation.


    * **num_epoch** (*int**, **optional*) – Current epoch number. Defaults to 1.


    * **hyperparam** (*bool**, **optional*) – If True enter Hypertuning mode. Defaults to False.


    * **log_history** (*bool**, **optional*) – If True log history. Defaults to False.



* **Returns**

    accuracy and loss.
    list: Log History.



* **Return type**

    dict



### class sudoai.trainer.TokenClassificationTrainer(id: str, model: sudoai.models.word.Word2Label = None, version: str = '0.1.0', teacher_forcing_ratio: float = 0.5, hidden_size: int = 512, print_every: int = 30000, lr: float = 0.0001, epochs: int = 2, drop_out: float = 0.1, do_eval: bool = False, do_save: bool = False, do_shuffle: bool = True, split_ratio: float = 0.2, do_save_checkpoint: bool = False, continue_from_checkpoint: str = None, loss: str = 'nll', optimizer: str = 'rmsprop', momentum: float = 0.0, wandb: <module 'wandb' from 'c:\\\\anaconda3\\\\lib\\\\site-packages\\\\wandb\\\\__init__.py'> = None, wandb_key: str = None, base_path: str = None)
Bases: `sudoai.trainer.core.Trainer`

Word to Label Trainer class.

Trainer for Word to label model.


#### id()
Dataset identifier.


* **Type**

    str



#### model()
Word To Label Model. Defaults to None.


* **Type**

    `Word2Label`, optional



#### version()
Model version. Defaults to ‘0.1.0’.


* **Type**

    str, optional



#### teacher_forcing_ratio()
Teacher Forcing ratio for acceleration. Defaults to 0.5.


* **Type**

    float, optional



#### hidden_size()
Hidden size of the model. Defaults to 512.


* **Type**

    int, optional



#### print_every()
Log frequency result by steps (log result every 30000 steps). Defaults to 30000.


* **Type**

    int, optional



#### lr()
Learn rate value. Defaults to 0.0001.


* **Type**

    float, optional



#### epochs()
Number of epochs. Defaults to 2.


* **Type**

    int, optional



#### drop_out()
Drop out value. Defaults to 0.1.


* **Type**

    float, optional



#### do_eval()
If True evaluate. Defaults to False.


* **Type**

    bool, optional



#### do_save()
If True save the model when training ends. Defaults to False.


* **Type**

    bool, optional



#### do_shuffle()
If True shuffle the dataset. Defaults to True.


* **Type**

    bool, optional



#### split_ratio()
Split ratio (0.2 meaning 80% training and 20% validation). Defaults to 0.2.


* **Type**

    float, optional



#### do_save_checkpoint()
If True save checkpoint every epoch. Defaults to False.


* **Type**

    bool, optional



#### continue_from_checkpoint()
Path of checkpoint as start point. Defaults to None.


* **Type**

    str, optional



#### loss()
Loss function. Defaults to ‘nll’.


* **Type**

    str, optional



#### optimizer()
Optimizer function. Defaults to ‘rmsprop’.


* **Type**

    str, optional



#### momentum()
Momentum value for optimizer. Defaults to 0.0.


* **Type**

    float, optional



#### wandb()
Wandb class to track logs. Defaults to None.


* **Type**

    wandb, optional



#### wandb_key()
Wandb api key. Defaults to None.


* **Type**

    str, optional



#### base_path()
Base path to save checkpoint and model. Defaults to None.


* **Type**

    str, optional



#### data_process(test_mode: bool = False)
Process dataset.

Split data for training and evaluation.
:param test_mode: If True enter in test mode. Defaults to False.
:type test_mode: bool, optional


* **Raises**

    **Exception** – when dataset is too small.



* **Returns**

    Dict with data for training and evaluation.



* **Return type**

    dict



#### evaluate(n=10)
Visual evaluation process for random input from dataset


* **Parameters**

    **n** (*int**, **optional*) – number of item to evaluate. Defaults to 10.



#### save()
Save model with id and version.


* **Returns**

    Path of model saved to.



* **Return type**

    str



#### save_checkpoint(num_epoch: int, train_loss_avg: float, eval_loss_avg: Optional[float] = None)
Save checkpoint


* **Parameters**

    
    * **num_epoch** (*int*) – Current epoch number.


    * **train_loss_avg** (*float*) – Current average training loss .


    * **eval_loss_avg** (*float**, **optional*) – Current average evaluation loss. Defaults to None.



#### start(hyperparam: bool = False, test_mode: bool = False, log_history: bool = False)
Entry point to begin the training process.


* **Parameters**

    
    * **hyperparam** (*bool**, **optional*) – If True process in HyperParameter mode. Defaults to False.


    * **test_mode** (*bool**, **optional*) – If True process in test mode


    * **mode to try with small amount of data****)**** Defaults to False.** (*(**Test*) – 


    * **log_history** (*bool**, **optional*) – If True logs all steps for reporting. Defaults to False.


**WARNING**: If your dataset is too small (less then 100) and you chose test mode,
an Exception raised.


* **Raises**

    **ModelError** – when model is None.



* **Returns**

    accuracy and loss.
    list: Log History.



* **Return type**

    dict



#### steps(train: dict, num_epoch: int = 1, hyperparam: bool = False, log_history: bool = False)
Training steps


* **Parameters**

    
    * **train** (*dict*) – Dict with data for training and evaluation.


    * **num_epoch** (*int**, **optional*) – Current epoch number. Defaults to 1.


    * **hyperparam** (*bool**, **optional*) – If True enter Hypertuning mode. Defaults to False.


    * **log_history** (*bool**, **optional*) – If True log history. Defaults to False.



* **Returns**

    accuracy and loss.
    list: Log History.



* **Return type**

    dict



### class sudoai.trainer.Trainer(id: str, version: str = '0.1.0', teacher_forcing_ratio: float = 0.5, hidden_size: int = 512, print_every: int = 30000, lr: float = 0.0001, epochs: int = 2, drop_out: float = 0.1, do_eval: bool = False, do_save: bool = False, do_shuffle: bool = True, split_ratio: float = 0.2, do_save_checkpoint: bool = False, continue_from_checkpoint: str = None, loss: str = 'nll', optimizer: str = 'rmsprop', momentum: float = 0.0, wandb: <module 'wandb' from 'c:\\\\anaconda3\\\\lib\\\\site-packages\\\\wandb\\\\__init__.py'> = None, wandb_key: str = None, base_path: str = None)
Bases: `object`

The base class sudoai.trainer.Trainer representing a Trainer.

All Trainers that represent a training process should subclass it.
All subclasses should overwrite __init__(), supporting the new attributes.
Subclasses could also optinally overwrite steps(), which is the main process
for training steps.


#### id()
Dataset identifier.


* **Type**

    str



#### version()
Model version. Defaults to ‘0.1.0’.


* **Type**

    str, optional



#### teacher_forcing_ratio()
Teacher Forcing ratio for acceleration. Defaults to 0.5.


* **Type**

    float, optional



#### hidden_size()
Hidden size of the model. Defaults to 512.


* **Type**

    int, optional



#### print_every()
Log frequency result by steps (log result every 30000 steps). Defaults to 30000.


* **Type**

    int, optional



#### lr()
Learn rate value. Defaults to 0.0001.


* **Type**

    float, optional



#### epochs()
Number of epochs. Defaults to 2.


* **Type**

    int, optional



#### drop_out()
Drop out value. Defaults to 0.1.


* **Type**

    float, optional



#### do_eval()
If True evaluate. Defaults to False.


* **Type**

    bool, optional



#### do_save()
If True save the model when training ends. Defaults to False.


* **Type**

    bool, optional



#### do_shuffle()
If True shuffle the dataset. Defaults to True.


* **Type**

    bool, optional



#### split_ratio()
Split ratio (0.2 meaning 80% training and 20% validation). Defaults to 0.2.


* **Type**

    float, optional



#### do_save_checkpoint()
If True save checkpoint every epoch. Defaults to False.


* **Type**

    bool, optional



#### continue_from_checkpoint()
Path of checkpoint as start point. Defaults to None.


* **Type**

    str, optional



#### loss()
Loss function. Defaults to ‘nll’.


* **Type**

    str, optional



#### optimizer()
Optimizer function. Defaults to ‘rmsprop’.


* **Type**

    str, optional



#### momentum()
Momentum value for optimizer. Defaults to 0.0.


* **Type**

    float, optional



#### wandb()
Wandb class to track logs. Defaults to None.


* **Type**

    wandb, optional



#### wandb_key()
Wandb api key. Defaults to None.


* **Type**

    str, optional



#### base_path()
Base path to save checkpoint and model. Defaults to None.


* **Type**

    str, optional



#### data_process(test_mode: bool = False)
Process dataset.

Split data for training and evaluation.
:param test_mode: If True enter in test mode. Defaults to False.
:type test_mode: bool, optional


* **Raises**

    **Exception** – when dataset is too small.



* **Returns**

    Dict with data for training and evaluation.



* **Return type**

    dict



#### evaluate(n=10)
Visual evaluation process for random input from dataset


* **Parameters**

    **n** (*int**, **optional*) – number of item to evaluate. Defaults to 10.



#### save()
Save model with id and version.


* **Returns**

    Path of model saved to.



* **Return type**

    str



#### save_checkpoint(num_epoch: int, train_loss_avg: float, eval_loss_avg: Optional[float] = None)
Save checkpoint


* **Parameters**

    
    * **num_epoch** (*int*) – Current epoch number.


    * **train_loss_avg** (*float*) – Current average training loss .


    * **eval_loss_avg** (*float**, **optional*) – Current average evaluation loss. Defaults to None.



#### start(hyperparam: bool = False, test_mode: bool = False, log_history: bool = False)
Entry point to begin the training process.


* **Parameters**

    
    * **hyperparam** (*bool**, **optional*) – If True process in HyperParameter mode. Defaults to False.


    * **test_mode** (*bool**, **optional*) – If True process in test mode


    * **mode to try with small amount of data****)**** Defaults to False.** (*(**Test*) – 


    * **log_history** (*bool**, **optional*) – If True logs all steps for reporting. Defaults to False.


**WARNING**: If your dataset is too small (less then 100) and you chose test mode,
an Exception raised.


* **Raises**

    **ModelError** – when model is None.



* **Returns**

    accuracy and loss.
    list: Log History.



* **Return type**

    dict



#### steps(train: dict, num_epoch: int = 1, hyperparam: bool = False, log_history: bool = False)
Training steps


* **Parameters**

    
    * **train** (*dict*) – Dict with data for training and evaluation.


    * **num_epoch** (*int**, **optional*) – Current epoch number. Defaults to 1.


    * **hyperparam** (*bool**, **optional*) – If True enter Hypertuning mode. Defaults to False.


    * **log_history** (*bool**, **optional*) – If True log history. Defaults to False.



* **Returns**

    accuracy and loss.
    list: Log History.



* **Return type**

    dict



### class sudoai.trainer.Word2WordTrainer(id: str, model: sudoai.models.word.Word2Word = None, version: str = '0.1.0', teacher_forcing_ratio: float = 0.5, hidden_size: int = 512, print_every: int = 30000, lr: float = 0.0001, epochs: int = 2, drop_out: float = 0.1, do_eval: bool = False, do_save: bool = False, do_shuffle: bool = True, split_ratio: float = 0.2, do_save_checkpoint: bool = False, continue_from_checkpoint: str = None, loss: str = 'nll', optimizer: str = 'rmsprop', momentum: float = 0.0, wandb: <module 'wandb' from 'c:\\\\anaconda3\\\\lib\\\\site-packages\\\\wandb\\\\__init__.py'> = None, wandb_key: str = None, base_path: str = None)
Bases: `sudoai.trainer.core.Trainer`

Word to Word Trainer class.

Trainer for word to word model.


#### id()
Dataset identifier.


* **Type**

    str



#### model()
Word To Word model. Defaults to None.


* **Type**

    `Word2Word`, optional



#### version()
Model version. Defaults to ‘0.1.0’.


* **Type**

    str, optional



#### teacher_forcing_ratio()
Teacher Forcing ratio for acceleration. Defaults to 0.5.


* **Type**

    float, optional



#### hidden_size()
Hidden size of the model. Defaults to 512.


* **Type**

    int, optional



#### print_every()
Log frequency result by steps (log result every 30000 steps). Defaults to 30000.


* **Type**

    int, optional



#### lr()
Learn rate value. Defaults to 0.0001.


* **Type**

    float, optional



#### epochs()
Number of epochs. Defaults to 2.


* **Type**

    int, optional



#### drop_out()
Drop out value. Defaults to 0.1.


* **Type**

    float, optional



#### do_eval()
If True evaluate. Defaults to False.


* **Type**

    bool, optional



#### do_save()
If True save the model when training ends. Defaults to False.


* **Type**

    bool, optional



#### do_shuffle()
If True shuffle the dataset. Defaults to True.


* **Type**

    bool, optional



#### split_ratio()
Split ratio (0.2 meaning 80% training and 20% validation). Defaults to 0.2.


* **Type**

    float, optional



#### do_save_checkpoint()
If True save checkpoint every epoch. Defaults to False.


* **Type**

    bool, optional



#### continue_from_checkpoint()
Path of checkpoint as start point. Defaults to None.


* **Type**

    str, optional



#### loss()
Loss function. Defaults to ‘nll’.


* **Type**

    str, optional



#### optimizer()
Optimizer function. Defaults to ‘rmsprop’.


* **Type**

    str, optional



#### momentum()
Momentum value for optimizer. Defaults to 0.0.


* **Type**

    float, optional



#### wandb()
Wandb class to track logs. Defaults to None.


* **Type**

    wandb, optional



#### wandb_key()
Wandb api key. Defaults to None.


* **Type**

    str, optional



#### base_path()
Base path to save checkpoint and model. Defaults to None.


* **Type**

    str, optional



#### data_process(test_mode: bool = False)
Process dataset.

Split data for training and evaluation.
:param test_mode: If True enter in test mode. Defaults to False.
:type test_mode: bool, optional


* **Raises**

    **Exception** – when dataset is too small.



* **Returns**

    Dict with data for training and evaluation.



* **Return type**

    dict



#### evaluate(n=10)
Visual evaluation process for random input from dataset


* **Parameters**

    **n** (*int**, **optional*) – number of item to evaluate. Defaults to 10.



#### save()
Save model with id and version.


* **Returns**

    Path of model saved to.



* **Return type**

    str



#### save_checkpoint(num_epoch: int, train_loss_avg: float, eval_loss_avg: Optional[float] = None)
Save checkpoint


* **Parameters**

    
    * **num_epoch** (*int*) – Current epoch number.


    * **train_loss_avg** (*float*) – Current average training loss .


    * **eval_loss_avg** (*float**, **optional*) – Current average evaluation loss. Defaults to None.



#### start(hyperparam: bool = False, test_mode: bool = False, log_history: bool = False)
Entry point to begin the training process.


* **Parameters**

    
    * **hyperparam** (*bool**, **optional*) – If True process in HyperParameter mode. Defaults to False.


    * **test_mode** (*bool**, **optional*) – If True process in test mode


    * **mode to try with small amount of data****)**** Defaults to False.** (*(**Test*) – 


    * **log_history** (*bool**, **optional*) – If True logs all steps for reporting. Defaults to False.


**WARNING**: If your dataset is too small (less then 100) and you chose test mode,
an Exception raised.


* **Raises**

    **ModelError** – when model is None.



* **Returns**

    accuracy and loss.
    list: Log History.



* **Return type**

    dict



#### steps(train: dict, num_epoch: int = 1, hyperparam: bool = False, log_history: bool = False)
Training steps


* **Parameters**

    
    * **train** (*dict*) – Dict with data for training and evaluation.


    * **num_epoch** (*int**, **optional*) – Current epoch number. Defaults to 1.


    * **hyperparam** (*bool**, **optional*) – If True enter Hypertuning mode. Defaults to False.


    * **log_history** (*bool**, **optional*) – If True log history. Defaults to False.



* **Returns**

    accuracy and loss.
    list: Log History.



* **Return type**

    dict
