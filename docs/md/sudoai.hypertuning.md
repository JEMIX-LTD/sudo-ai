# Hypertuning

## Module Content

Hypertuning module
@author: Aymen Jemi (jemix) <[jemiaymen@gmail.com](mailto:jemiaymen@gmail.com)> at SUDO-AI

sudoai.hypertuning is a lightweight and extensible library to easily
try different parameters for Natural Language Processing (NLP) model.

**WARNING**: To try hypertuning for dataset, it must be exist in
drive or local storage.

### Examples

These examples illustrate how to use hypertuning modules.

Word to Label classification:

```python
>>> token_hypertuning(dataset_id='w2l_dataset_model', test_mode=True, n_experience=2)
```

Word to Word model.

```python
>>> w2w_hypertuning('t_dataset_model', n_experience=2, test_mode=True)
```


### sudoai.hypertuning.seq2label_hypertuning(dataset_id: str, n_experience: int = 30, dataset_version: str = '0.1.0', test_mode: bool = False)
Hyper tuning method for Seq2Label model.


* **Parameters**

    
    * **dataset_id** (*str*) – Dataset unique id.


    * **n_experience** (*int**, **optional*) – Number of experiences. Defaults to 30.


    * **dataset_version** (*str**, **optional*) – Dataset version. Defaults to ‘0.1.0’.


    * **test_mode** (*bool**, **optional*) – Test mode to try with small amount of data. Defaults to False.


**NOTE**: For more information check quickstart docs [http://sudoai.tech/quickstart](http://sudoai.tech/quickstart)


* **Returns**

    (best_parameters, metrics).



* **Return type**

    tuple



### sudoai.hypertuning.token_hypertuning(dataset_id: str, n_experience: int = 30, dataset_version: str = '0.1.0', test_mode: bool = False)
Hyper tuning method for token classification (word2label) model.


* **Parameters**

    
    * **dataset_id** (*str*) – Dataset unique id.


    * **n_experience** (*int**, **optional*) – Number of experiences. Defaults to 30.


    * **dataset_version** (*str**, **optional*) – Dataset version. Defaults to ‘0.1.0’.


    * **test_mode** (*bool**, **optional*) – Test mode to try with small amount of data. Defaults to False.


**NOTE**: For more information check quickstart docs [http://sudoai.tech/quickstart](http://sudoai.tech/quickstart)


* **Returns**

    (best_parameters, metrics).



* **Return type**

    tuple



### sudoai.hypertuning.w2w_hypertuning(dataset_id: str, n_experience: int = 30, dataset_version: str = '0.1.0', test_mode: bool = False)
Hyper tuning method for (word2word) model.


* **Parameters**

    
    * **dataset_id** (*str*) – Dataset unique id.


    * **n_experience** (*int**, **optional*) – Number of experiences. Defaults to 30.


    * **dataset_version** (*str**, **optional*) – Dataset version. Defaults to ‘0.1.0’.


    * **test_mode** (*bool**, **optional*) – Test mode to try with small amount of data. Defaults to False.


**NOTE**: For more information check quickstart docs [http://sudoai.tech/quickstart](http://sudoai.tech/quickstart)


* **Returns**

    (best_parameters, metrics).



* **Return type**

    tuple
