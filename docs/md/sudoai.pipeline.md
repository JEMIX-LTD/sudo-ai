# Pipeline

## Module Content

Pipeline module
@author: Aymen Jemi (jemix) <[jemiaymen@gmail.com](mailto:jemiaymen@gmail.com)> at SUDO-AI

Pipeline for natural language processing, sudoai.pipeline
is a lightweight library to run easily any model.

### Examples

These examples illustrate how to use sudoai Pipeline.

Create and save pipeline:

```python
>>> config = PipelineConfig(id='lid',model_type=ModelType.FASTTEST)
>>> nlp = Pipeline(config=config, compressed=True)
>>> nlp(inputs='chbik blid sa7ebi')
'aeb'
>>> nlp(inputs='I ask always and no one answer me')
'eng'
>>> nlp.save(True)
```


### class sudoai.pipeline.ModelType(value)
Bases: `str`, `enum.Enum`

Enumerate for models types, accepted by Pipeline class.


#### WORD_TO_LABEL()
Word2Label model (1) .


* **Type**

    str



#### WORD_TO_WORD()
Word2Label model (2).


* **Type**

    str



#### SEQ_TO_LABEL()
Seq2Label model (3).


* **Type**

    str



#### SEQ_TO_SEQ()
For Seq2Seq model (4).


* **Type**

    str



#### FASTTEXT()
FastText model (5).


* **Type**

    str



#### FASTTEXT(: str = '5')

#### SEQ_TO_LABEL(: str = '3')

#### SEQ_TO_SEQ(: str = '4')

#### WORD_TO_LABEL(: str = '1')

#### WORD_TO_WORD(: str = '2')

### class sudoai.pipeline.Pipeline(config: Optional[sudoai.pipeline.core.PipelineConfig] = None, \*\*kwargs)
Bases: `object`

Pipeline with a `PipelineConfig`.

Create nlp pipeline with pipeline config file.


#### config()
Pipeline configuration class.


* **Type**

    `PipelineConfig`



#### model()
Current model.


* **Type**

    Model



#### src_tokenizer()
Source tokenizer.


* **Type**

    Tokenizer



#### target_tokenizer()
Target tokenizer.


* **Type**

    Tokenizer



#### tokenizer()
Tokenizer.


* **Type**

    Tokenizer



#### io()
To Download models and tokenizer if not exists.


* **Type**

    `InputOutput`



#### kwargs()
Additional arguments.


* **Type**

    Dict(str,any)



#### load()
Load model and tokenizer from Pipeline config.


* **Raises**

    
    * **PipelineException** – When config file not found.


    * **PipelineException** – When model is None.


    * **PipelineException** – When tokenizer(s) is None.



#### predict(inputs)
Predict method.


* **Parameters**

    **inputs** (*str*) – 



* **Raises**

    **PipelineException** – When inputs is None.



#### save(override: bool = False)
Save Pipeline.


* **Parameters**

    **override** (*bool**, **optional*) – If pipeline config file exist ecrase it. Defaults to False.



### class sudoai.pipeline.PipelineConfig(id: Optional[str] = None, version: str = '0.1.0', description: str = 'default description', i2l: Dict[int, str] = <factory>, is_two_tokenizer: bool = False, model_type: sudoai.pipeline.core.ModelType = <ModelType.WORD_TO_LABEL: '1'>)
Bases: `object`

This is a data class definition, to identify all pipeline info.

A `PipelineConfig` class definition contains identificaiton and
allocation information for `Pipeline`.


#### id()
Identifier for Model and Tokenizer(s).


* **Type**

    str



#### version()
Model and Pipeline version. Defaults to ‘0.1.0’


* **Type**

    str, optional



#### desscription()
Pipeline description. Defaults to ‘default description’


* **Type**

    str, optional



#### is_two_tokenizer()
If model have two tokenizer (word2word). Defaults to False


* **Type**

    bool, optional



#### model_type()
Model type. Defaults to ‘WORD_TO_LABEL’


* **Type**

    `ModelType`, optional



#### i2l()
Index to label dict. Defaults to {0: “bad”, 1: “good”}


* **Type**

    Dict[int, str], optional



#### description(: str = 'default description')

#### i2l(: Dict[int, str])

#### id(: str = None)

#### is_two_tokenizer(: bool = False)

#### model_type(: sudoai.pipeline.core.ModelType = '1')

#### version(: str = '0.1.0')

### exception sudoai.pipeline.PipelineException(reason: str, msg='pipeline error')
Bases: `Exception`

Exception raised in Pipeline class.


* **Parameters**

    
    * **reason** (*str*) – Reason why pipeline not working.


    * **msg** (*str**, **optional*) – Human readable string describing the exception. Default to ‘pipeline error’



#### reason()
Reason why pipeline not working.


* **Type**

    str



#### msg()
Human readable string describing the exception.


* **Type**

    str



#### args()

#### with_traceback()
Exception.with_traceback(tb) –
set self.__traceback__ to tb and return self.


### sudoai.pipeline.load_pipeline_config(id: str, version: str = '0.1.0')
Load pipeline config from local storage with id and version.


* **Parameters**

    
    * **id** (*str*) – Pipeline config identifier.


    * **version** (*str**, **optional*) – Pipeline version. Defaults to ‘0.1.0’.



* **Raises**

    
    * **FileNotFoundError** – When pipeline config file not exist.


    * **TypeError** – When pipeline config file is not correct.



* **Returns**

    Saved pipeline config.



* **Return type**

    `PipelineConfig`



### sudoai.pipeline.predict_from_ft(inputs)
Normalize predicted result from fasttext.


* **Parameters**

    **inputs** (*tuple**(**tuple**(**str**)**,**tuple**(**float**)**)*) – FastText predict result.



* **Returns**

    Predicted label.
    list: List of predicted labels.



* **Return type**

    str



### sudoai.pipeline.save_pipeline_config(config: sudoai.pipeline.core.PipelineConfig, override=False)
Save pipeline config in local storage with id and version.


* **Parameters**

    
    * **config** (*PipelineConfig*) – Pipeline config to save it.


    * **override** (*bool**, **optional*) – If pipeline config file exist ecrase it. Defaults to False.



* **Raises**

    **FileExistsError** – When pipeline config file exist and override is False.
