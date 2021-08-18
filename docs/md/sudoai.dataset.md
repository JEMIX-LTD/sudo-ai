# Dataset

## Module Content

Dataset module
@author: Aymen Jemi (jemix) <[jemiaymen@gmail.com](mailto:jemiaymen@gmail.com)> at SUDO-AI

Datasets for natural language processing, sudoai.dataset
is a lightweight and extensible library to easily share
and access datasets for Natural Language Processing (NLP).

### Examples

These examples illustrate how to use DataType in DatasetInfo class.

Text data file:

```python
>>> info = DatasetInfo(id='sa', data_path='data.txt', data_type=DataType.TEXT)
>>> info.data_type
<DataType.TEXT: '1'>
```

Excel data file:

```python
>>> info = DatasetInfo(id='sa', data_path='data.xlsx', data_type=DataType.EXCEL)
>>> info.data_type
<DataType.TEXT: '4'>
```


### class sudoai.dataset.DataType(value)
Bases: `str`, `enum.Enum`

Enumerate for accepted data types by Dataset class.


#### TEXT()
For text data file (1).


* **Type**

    str



#### CSV()
For csv data file (2).


* **Type**

    str



#### JSON()
For json data file (3).


* **Type**

    str



#### EXCEL()
For excel data file (4).


* **Type**

    str


### Examples

These examples illustrate how to use DataType in DatasetInfo class.

Text data file:

```python
>>> info = DatasetInfo(id='sa', data_path='data.txt', data_type=DataType.TEXT)
>>> info.data_type
<DataType.TEXT: '1'>
```

Excel data file:

```python
>>> info = DatasetInfo(id='sa', data_path='data.xlsx', data_type=DataType.EXCEL)
>>> info.data_type
<DataType.TEXT: '4'>
```


#### CSV( = '2')

#### EXCEL( = '4')

#### JSON( = '3')

#### TEXT( = '1')

### class sudoai.dataset.Dataset(info: sudoai.dataset.core.DatasetInfo, \*\*kwargs)
Bases: `object`

Defines a dataset with DatasetInfo.

Create a dataset from a data file (text, csv, json or excel) .
Don’t forget to attribute unique identifian to DatasetInfo.id .


#### info()
DatasetInfo with all dataset information .


* **Type**

    `DatasetInfo`



#### data()
List of encoded tokens in torch.tensor .


* **Type**

    `list`



#### plain_data()
List of plain text in str.


* **Type**

    `list`


### Examples

These examples illustrate how to use Dataset class.

Word2Label model.

```python
>>> info = DatasetInfo(id='sa', data_path='data.txt', dataset_type=DatasetType.WORD_TO_LABEL)
>>> tokenizer = load_tokenizer('sa')
>>> dataset = Dataset(info=info,tokenizer=tokenizer)
```

Word2Word model.

```python
>>> info = DatasetInfo(id='sa', data_path='data.txt', dataset_type=DatasetType.WORD_TO_WORD)
>>> t_src = load_tokenizer('sa/src')
>>> t_target = load_tokenizer('sa/target')
>>> dataset = Dataset(info=info, src_tokenizer=t_src, target_tokenizer=t_target)
```


#### build()
Build dataset .

**NOTE**: If dataset_type is TEXT data contains str.

If dataset_type is WORD_TO_LABEL or
SEQ_TO_LABEL data contains Tuple(str, label).

If dataset_type is WORD_TO_WORD or SEQ_TO_SEQ data contains Tuple(str, str).


* **Returns**

    None.



#### data_to_tensor(data: str)
Transform data to tokens and return tensor with index from tokens.


* **Parameters**

    **data** (*str*) – Line of data in string format.



* **Returns**

    Torch Tensor with tokens index.



* **Return type**

    torch.Tensor



#### label_to_tensor(label: str)
Transform label to index and return tensor.


* **Parameters**

    **label** (str) – Label in string format.



* **Raises**

    **TypeError** – If label not found in self.info.l2i .



* **Returns**

    Torch Tensor with label index.



* **Return type**

    torch.Tensor



#### classmethod load(id: str)
Load saved dataset.


* **Parameters**

    **id** (*str*) – Dataset unique id.



* **Returns**

    Dataset from id.



* **Return type**

    `Dataset`



#### n_class()
Get number of label classes.


* **Returns**

    Number of label classes.



* **Return type**

    int



#### save(override: bool = False, is_ziped: bool = False, algo: sudoai.utils.datasets.ZipAlgo = <ZipAlgo.LZMA: '3'>)
Save dataset with id and version and compressed algo if exist.


* **Parameters**

    
    * **override** (*bool**, **optional*) – If True delete old dataset file. Defaults to False.


    * **is_ziped** (*bool**, **optional*) – If True compress dataset file. Defaults to False.


    * **algo** (`ZipAlgo`, optional) – Compressing algorithm (LZMA, BZ2 or GZIP). Defaults to ZipAlgo.LZMA.



* **Raises**

    **FileExistsError** – If override is False and dataset file exists.


**WARNING**: If you set is_ziped True you must chose `ZipAlgo`.


* **Returns**

    True if dataset saved.



* **Return type**

    bool



### exception sudoai.dataset.DatasetError(dataset, message='Dataset error!!')
Bases: `Exception`

Exception raised in Dataset class.


* **Parameters**

    
    * **dataset** (`Dataset`) – Current dataset object.


    * **message** (*str**, **optional*) – Human readable string describing the exception. Defaults to “Dataset error!!”.



#### dataset()
Current dataset object.


* **Type**

    `Dataset`



#### message()
Human readable string describing the exception.


* **Type**

    str



#### args()

#### with_traceback()
Exception.with_traceback(tb) –
set self.__traceback__ to tb and return self.


### class sudoai.dataset.DatasetInfo(id: str, data_path: str, version: str = '0.1.0', description: Optional[str] = None, str_text: str = 'text', str_label: str = 'label', i2l: Dict[int, str] = <factory>, l2i: Dict[str, list] = <factory>, data_type: sudoai.dataset.core.DataType = <DataType.TEXT: '1'>, dataset_type: sudoai.dataset.core.DatasetType = <DatasetType.TEXT: '1'>, sep: str = '\\t', sep_label: str = ', ', encoding: str = 'utf8', verbose: int = 1, max_length: int = 256, do_lower_case: bool = True, strip_duplicated: bool = False, strip_punc: bool = False, stopword: Optional[sudoai.preprocess.text.StopWord] = None)
Bases: `object`

This is a data class definition, to identify all info in dataset.

> A DatasetInfo class definition contains identification and
> allocation information for Dataset.
> Each DatasetInfo class must contains unique id that contains
> [a-zA-Z0-9] characters without spaces.


#### id()
Dataset identifier.


* **Type**

    str



#### data_path()
Path of data file.


* **Type**

    str



#### version()
Dataset version. Defaults to “0.1.0”.


* **Type**

    str, optional



#### description()
Dataset description. Defaults to None.


* **Type**

    str, optional



#### str_text()
Column name of text (json, excel or csv). Default to ‘text’.


* **Type**

    str, optional



#### str_label()
Column name of label (json, excel or csv). Default to ‘label’.


* **Type**

    str, optional



#### data_type()
Enumerate for dataset types. Default to DataType.TEXT.


* **Type**

    DataType, optional



#### dataset_type()
Enumerate for dataset types. Defaults to DatasetType.TEXT.


* **Type**

    DatasetType, optional



#### sep()
Separator between text and labels. Defaults to ‘   ‘.


* **Type**

    str, optional



#### sep_label()
Separator between labels. Defaults to ‘,’.


* **Type**

    str, optional



#### encoding()
Text encoding. Defaults to ‘utf8’.


* **Type**

    str, optional



#### do_lower_case()
For lowercase text. Defaults to True.


* **Type**

    bool, optional



#### verbose()
Verbose value. Defaults to 1.


* **Type**

    int, optional



#### max_length()
Number of max elements (tokens). Defaults to 256.


* **Type**

    int, optional



#### strip_duplicated()
To strip duplicated lettre (exp hhhhello ~> hello). Defaults to False.


* **Type**

    bool, optional



#### strip_punc()
To strip punctuation. Defaults to False.


* **Type**

    bool, optional



#### stopword()
StopWord class to espace stopwords. Defaults to None.


* **Type**

    StopWord, optional



#### l2i()
Convert label to index. Defaults to {0:’bad’} .


* **Type**

    Dict[int, str], optional



#### i2l()
Convert index to label. Defaults to {‘good’:[1]} .


* **Type**

    Dict[str, list], optional


### Examples

These examples illustrate how to use DatasetInfo class.

```python
>>> info = DatasetInfo(id='sa', data_path='data.xlsx', data_type=DataType.EXCEL)
```

```python
>>> info = DatasetInfo(id='sa', data_path='data.txt',dataset_type=DatasetType.WORD_TO_LABEL)
```


#### data_path(: str)

#### data_type(: sudoai.dataset.core.DataType = '1')

#### dataset_type(: sudoai.dataset.core.DatasetType = '1')

#### description(: str = None)

#### do_lower_case(: bool = True)

#### encoding(: str = 'utf8')

#### i2l(: Dict[int, str])

#### id(: str)

#### l2i(: Dict[str, list])

#### max_length(: int = 256)

#### sep(: str = '\\t')

#### sep_label(: str = ',')

#### stopword(: sudoai.preprocess.text.StopWord = None)

#### str_label(: str = 'label')

#### str_text(: str = 'text')

#### strip_duplicated(: bool = False)

#### strip_punc(: bool = False)

#### verbose(: int = 1)

#### version(: str = '0.1.0')

### class sudoai.dataset.DatasetType(value)
Bases: `str`, `enum.Enum`

Enumerate for dataset types, to work with sudoai.models accepted by Dataset class.


#### TEXT()
For Text model (1) .


* **Type**

    str



#### WORD_TO_LABEL()
For Word2Label model (2).


* **Type**

    str



#### WORD_TO_WORD()
For Word2Word model (3).


* **Type**

    str



#### SEQ_TO_LABEL()
For Seq2Label model (4).


* **Type**

    str



#### SEQ_TO_SEQ()
For Seq2Seq model (4).


* **Type**

    str


### Examples

These examples illustrate how to use DatasetType in DatasetInfo class.

For Word2Label model:

```python
>>> info = DatasetInfo(id='sa', data_path='data.xlsx',dataset_type=DatasetType.WORD_TO_LABEL)
>>> info.dataset_type
<DatasetType.WORD_TO_LABEL: '2'>
```

For Seq2Label:

```python
>>> info = DatasetInfo(id='sa', data_path='data.txt',dataset_type=DatasetType.SEQ_TO_LABEL)
>>> info.dataset_type
<DatasetType.WORD_TO_LABEL: '4'>
```


#### SEQ_TO_LABEL( = '4')

#### SEQ_TO_SEQ( = '5')

#### TEXT( = '1')

#### WORD_TO_LABEL( = '2')

#### WORD_TO_WORD( = '3')
