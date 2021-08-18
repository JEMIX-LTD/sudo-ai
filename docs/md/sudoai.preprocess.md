# Preprocess

## Module Content

Preprocess.Text module
@author: Aymen Jemi (jemix) <[jemiaymen@gmail.com](mailto:jemiaymen@gmail.com)> at SUDO-AI

Preprocessing for natural language processing, sudoai.preprocess
is a lightweight library to clean and tokenize text.

### Examples

These examples illustrate how to use sudoai Preprocess module.

Create and save CharTokenizer:

```python
>>> tokenizer = CharTokenizer(strip_duplicated=True, do_lower_case=True)
>>> tokenizer.train(path="data.txt")
>>> tokenizer.save('id_chartokenizer', True)
train chartokenizer: 67534 lines [00:00, 79429.67 lines/s]
True
```

```python
>>> tokenizer('winek')
tensor([[ 2],
        [53],
        [20],
        [59],
        [18]], device='cuda:0')
```

```python
>>> en = tokenizer.encode('winek')
>>> tokenizer.decode(en)
['w', 'i', 'n', 'e', 'k']
```


### class sudoai.preprocess.BasicTokenizer(max_vocab_size: Optional[int] = None, do_lower_case: bool = True, strip_duplicated: bool = False, strip_punc: bool = False, strip_stop_words: bool = False, stopword: Optional[sudoai.preprocess.text.StopWord] = None)
Bases: `object`

The base class sudoai.preprocess.BasicTokenizer representing a Tokenizer.


#### lower_case()
If True lower case text.


* **Type**

    bool



#### duplicated()
If True clean text from duplicated characters.


* **Type**

    bool



#### punc()
If True clean text form punctuations.


* **Type**

    bool



#### max_size()
Maximum vocabulary size.


* **Type**

    int



#### vocab_size()
Vocabulary size.


* **Type**

    int



#### vocab()
Vocaburaly.


* **Type**

    set(str)



#### words_freq()
Words frequency.


* **Type**

    `Counter`



#### strip_stopword()
If True clean text from stopwords.


* **Type**

    bool



#### stopword()
Stopword class.


* **Type**

    `StopWord`



#### w2i()
Dictionary of word to index.


* **Type**

    dict(str, int)



#### i2w()
Dictionary of index to word.


* **Type**

    dict(int, str)


**WARNING**: If you set stip_stop_words True you should set stopword, if not
an Exception raised.


#### decode(ids: list)
Decode ids to text.


* **Parameters**

    **ids** (*list*) – Input ids.



* **Returns**

    Output words.



* **Return type**

    list



#### encode(text: str)
Encode text to ids.


* **Parameters**

    **text** (*str*) – Input text line.



* **Raises**

    **NotTrainedError** – When tokenizer not trained yet.



* **Returns**

    ids.



* **Return type**

    list



#### from_pretrain(vocab_file: str, words_freq: str)
Load tokenizer from saved vocab file and words frequency file.


* **Parameters**

    
    * **vocab_file** (*str*) – Path of vocabulary file.


    * **words_freq** (*str*) – Path of words frequency file.



* **Raises**

    
    * **FileExistsError** – When vocabulary file not exist.


    * **FileExistsError** – When words frequency file not exist.


    * **ValueError** – When words frequency file is corrupted.



#### load(id: str)
Load saved tokenizer.


* **Parameters**

    **id** (*str*) – Unique id of tokenizer.



* **Returns**

    Tokenizer .



* **Return type**

    Tokenizer



#### save(id: str, override: bool = False)
Save Tokenizer.


* **Parameters**

    
    * **id** (*str*) – Unique id of tokenizer.


    * **override** (*bool**, **optional*) – If True erase old tokenizer file. Defaults to False.



* **Raises**

    **FileExistsError** – When tokenizer file is exist and override is False.



* **Returns**

    Tokenizer saved or not.



* **Return type**

    bool



#### save_pretrain(vocab_file: str, words_freq: str)
Save vocabulary file and words frequency file.


* **Parameters**

    
    * **vocab_file** (*str*) – Path of vocabulary file.


    * **words_freq** (*str*) – Path of words frequency file.



#### split_on_punc(text: str)
Splits punctuation on a piece of text.


* **Returns**

    Tokens.



* **Return type**

    list



#### tokenize(text: str)
Tokenize text.


* **Parameters**

    **text** (*str*) – Input text line.



* **Returns**

    Tokens.



* **Return type**

    list



#### train(path: Optional[str] = None, paths: Optional[list] = None)
Train current Tokenizer.


* **Parameters**

    
    * **path** (*str**, **optional*) – Path of text data. Defaults to None.


    * **paths** (*list**, **optional*) – Paths of text data. Defaults to None.


### Notes

Split text into small pieces of text it’s take less time.
Big file always take more time then small pieces of file.


* **Raises**

    
    * **ValueError** – When path and paths is both None.


    * **TypeError** – When path or paths is wrong type.



#### word_frequency(path: str, word_freq: Optional[collections.Counter] = None)
Get word frequency from text file.


* **Parameters**

    
    * **path** (*str*) – Path of text data.


    * **word_freq** (`Counter`, optional) – Old Counter to update it with new Counter. Defaults to None.



* **Raises**

    **FileExistsError** – When data file not exist.



* **Returns**

    Words frequency.



* **Return type**

    `Counter`



### class sudoai.preprocess.CharTokenizer(do_lower_case: bool = True, strip_duplicated: bool = False, do_clean: bool = True, stopword: Optional[sudoai.preprocess.text.StopWord] = None)
Bases: `object`

Char Tokenizer class.


#### lower_case()
If True lower case text.


* **Type**

    bool



#### duplicated()
If True clean text from duplicated characters.


* **Type**

    bool



#### stopword()
Stopword class.


* **Type**

    `StopWord`



#### clean()
If True clean data text.


* **Type**

    bool



#### c2i()
Dictionary of character to index.


* **Type**

    dict(str, int)



#### i2c()
Dictionary of index to character.


* **Type**

    dict(int, str)



#### vocab_size()
Vocabulary size.


* **Type**

    int



#### vocab()
Vocaburaly.


* **Type**

    set(str)



#### _check_vocab()
Check vocabulary and fix error if exists.


#### chars(path: str)
Get unique characters from file and update current vocab.


* **Parameters**

    **path** (*str*) – Path of text data.



#### decode(ids: list)
Decode ids to characters.


* **Parameters**

    **ids** (*list*) – Input ids.



* **Returns**

    Output characters.



* **Return type**

    list



#### encode(text: str)
Encode text to ids.


* **Parameters**

    **text** (*str*) – Input text.



* **Raises**

    **NotTrainedError** – When tokenizer not trained yet.



* **Returns**

    ids.



* **Return type**

    list



#### from_pretrain(vocab_file: str)
Load character tokenizer from saved vocab file.


* **Parameters**

    **vocab_file** (*str*) – Path of vocabulary file.



* **Raises**

    **FileExistsError** – When vocabulary file not exist.



#### init(do_lower_case: bool, strip_duplicated: bool, do_clean: bool, stopword: sudoai.preprocess.text.StopWord)
Reinit attributes of CharTokenizer.


* **Parameters**

    
    * **do_lower_case** (*bool*) – If True lower case text.


    * **strip_duplicated** (*bool*) – If True clean text from duplicated characters.


    * **do_clean** (*bool*) – If True clean text.


    * **stopword** (`StopWord`) – Stopword class.



#### load(id: str)
Load saved character tokenizer.


* **Parameters**

    **id** (*str*) – Unique id of tokenizer.



* **Returns**

    Character tokenizer .



* **Return type**

    `CharTokenizer`



#### save(id: str, override: bool = False)
Save CharTokenizer.


* **Parameters**

    
    * **id** (*str*) – Unique id of tokenizer.


    * **override** (*bool**, **optional*) – If True erase old tokenizer file. Defaults to False.



* **Raises**

    **FileExistsError** – When tokenizer file is exist and override is False.



* **Returns**

    Tokenizer saved or not.



* **Return type**

    bool



#### save_pretrain(vocab_file: str)
Save vocabulary file.


* **Parameters**

    **vocab_file** (*str*) – Path of vocabulary file.



#### tokenize(text: str)
Character tokenize text.


* **Parameters**

    **text** (*str*) – Input text.



* **Returns**

    Tokens



* **Return type**

    list



#### train(path: Optional[str] = None, paths: Optional[list] = None)
Train character tokenizer.


* **Parameters**

    
    * **path** (*str**, **optional*) – Path of text data. Defaults to None.


    * **paths** (*list**, **optional*) – Paths of text data. Defaults to None.



* **Raises**

    
    * **ValueError** – When path and paths both are None.


    * **TypeError** – When path or paths is wrong type.



### exception sudoai.preprocess.InputTypeError(str_or_list, message='Is not str or list .')
Bases: `Exception`

Exception raised for errors in the input.


* **Parameters**

    
    * **str_or_list** (*str | list*) – input str_or_list which caused the error.


    * **message** (*str**, **optinal*) – Human readable string describing the exception. Defaults to ‘Is not str or list’.



#### str_or_list()
input str_or_list which caused the error.


* **Type**

    str | list



#### message()
Human readable string describing the exception.


* **Type**

    str



#### args()

#### with_traceback()
Exception.with_traceback(tb) –
set self.__traceback__ to tb and return self.


### exception sudoai.preprocess.NotTrainedError(obj, message='Is not trained yet')
Bases: `Exception`

Exception raised when tokenizer not trained yet.


* **Parameters**

    
    * **obj** (`BasicTokenizer`) – 


    * **message** (*str**, **optinal*) – Human readable string describing the exception. Defaults to ‘Is not trained yet’



#### obj()

* **Type**

    `BasicTokenizer`



#### message()
Human readable string describing the exception.


* **Type**

    str



#### args()

#### with_traceback()
Exception.with_traceback(tb) –
set self.__traceback__ to tb and return self.


### class sudoai.preprocess.PrefixSuffix(prefix=['ب', 'ك', 'س', 'و', 'ال', 'أ', 'ف', 'ل'], suffix=['ين', 'ان', 'و', 'ه', 'ك', 'ا', 'ي', 'ن', 'ت', 'ات', 'ون', 'وا', 'تم', 'هم', 'كم'])
Bases: `object`

Detect Prefix and Suffix .


#### suffix()
List of all suffix.


* **Type**

    list(str)



#### prefix()
List of all prefix.


* **Type**

    list(str)


### Examples

These examples illustrate how to use PrefixSuffix class.

```python
>>> clean = PrefixSuffix()
>>> clean('الهدف')
['##ال', 'هدف']
>>> clean('نمتم')
['نم', 'تم##']
```


#### get(token: str)
Get Prefix and Suffix.


* **Parameters**

    **token** (*str*) – Input token.



* **Returns**

    List with prefix (if exist) and suffix (if exist) and base.



* **Return type**

    list(str)



### class sudoai.preprocess.StopWord(path: Optional[str] = None, id: Optional[str] = None)
Bases: `object`

StopWord class to escape stopwords.


#### path()
Path of stopwords file.


* **Type**

    str



#### id()
Stopwords file id.


* **Type**

    str



#### words()
All stopwords.


* **Type**

    set(str)



#### is_stopword(word: str)
Check if word is a stopword.


* **Parameters**

    **word** (*str*) – Input word.


### Examples

These examples illustrate how to use StopWord class.

```python
>>> sw = StopWord(path='../data/ttd/src.txt')
>>> sw('alik')
True
>>> sw.is_stopword('not')
False
```


* **Returns**

    Result if is stopword or not.



* **Return type**

    bool



#### load()
Load stopwords with path or id.


* **Raises**

    
    * **FileNotFoundError** – When stopwords file not found.


    * **FileNotFoundError** – When stopwords id is wrong.


    * **Exception** – When stopwords path is None and id is None.



### class sudoai.preprocess.WordTokenizer(max_vocab_size: Optional[int] = None, do_lower_case: bool = True, strip_duplicated: bool = False, trip_punc: bool = False, strip_stop_words: bool = False, stopword: Optional[sudoai.preprocess.text.StopWord] = None)
Bases: `object`

Word Tokenizer class.


#### basic()
BasicTokenizer class.


* **Type**

    `BasicTokenizer`



#### prefix_suffix()

* **Type**

    `PrefixSuffix`



#### decode(ids: list)
Decode ids to text.


* **Parameters**

    **ids** (*list*) – Input ids.



* **Returns**

    Output words.



* **Return type**

    list



#### encode(text: str)
Encode text to ids.


* **Parameters**

    **text** (*str*) – Input text line.



* **Raises**

    **NotTrainedError** – When tokenizer not trained yet.



* **Returns**

    ids.



* **Return type**

    list



#### from_pretrain(vocab_file: str, words_freq: str)
Load tokenizer from saved vocab file and words frequency file.


* **Parameters**

    
    * **vocab_file** (*str*) – Path of vocabulary file.


    * **words_freq** (*str*) – Path of words frequency file.



* **Raises**

    
    * **FileExistsError** – When vocabulary file not exist.


    * **FileExistsError** – When words frequency file not exist.


    * **ValueError** – When words frequency file is corrupted.



#### init(do_lower_case: bool, strip_duplicated: bool, strip_punc: bool, stopword: sudoai.preprocess.text.StopWord)
Reinit attributes of WordTokenizer.


* **Parameters**

    
    * **do_lower_case** (*bool*) – If True lower case text.


    * **strip_duplicated** (*bool*) – If True clean text from duplicated characters.


    * **strip_punc** (*bool*) – If True clean text form punctuations.


    * **stopword** (`StopWord`) – Stopword class.



#### load(id: str)
Load saved word tokenizer.


* **Parameters**

    **id** (*str*) – Unique id of tokenizer.



* **Returns**

    Word tokenizer .



* **Return type**

    `WordTokenizer`



#### save(id: str, override: bool = False)
Save WordTokenizer.


* **Parameters**

    
    * **id** (*str*) – Unique id of tokenizer.


    * **override** (*bool**, **optional*) – If True erase old tokenizer file. Defaults to False.



* **Raises**

    **FileExistsError** – When tokenizer file is exist and override is False.



* **Returns**

    Tokenizer saved or not.



* **Return type**

    bool



#### save_pretrain(vocab_file: str, words_freq: str)
Save vocabulary file and words frequency file.


* **Parameters**

    
    * **vocab_file** (*str*) – Path of vocabulary file.


    * **words_freq** (*str*) – Path of words frequency file.



#### tokenize(text: str)
Tokenize text.


* **Parameters**

    **text** (*str*) – Input text line.



* **Returns**

    Tokens.



* **Return type**

    list



#### train(path: Optional[str] = None, paths: Optional[list] = None)
Train current Tokenizer.


* **Parameters**

    
    * **path** (*str**, **optional*) – Path of text data. Defaults to None.


    * **paths** (*list**, **optional*) – Paths of text data. Defaults to None.



### sudoai.preprocess.clean_text(text: str)
Clean text.


* **Parameters**

    **text** (*str*) – Input text.



* **Returns**

    Clean text.



* **Return type**

    str



### sudoai.preprocess.convert_to_unicode(text)
Convert input to unicode text.


* **Parameters**

    **text** (*str | bytes*) – Input data.



* **Raises**

    **ValueError** – When input text is not in (str, bytes)



* **Returns**

    Unicode text.



* **Return type**

    str



### sudoai.preprocess.strip_accents(text: str)
Clean text from accents.


* **Parameters**

    **text** (*str*) – Input data text.



* **Returns**

    Output text cleaned from accents.



* **Return type**

    str



### sudoai.preprocess.strip_duplicated_letter(text: str, tokens: bool = False)
Clean text from duplicated lettre.


* **Parameters**

    
    * **text** (*str*) – Input data text.


    * **tokens** (*bool**, **optional*) – If True return tokens not str. Defaults to False.


### Examples

These examples illustrate how to use strip_duplicated_lettre().

```python
>>> strip_duplicated_letter('helllllo word')
'hello word'
>>> strip_duplicated_letter('hellllllllo dude', True)
['hello', 'dude']
```


* **Returns**

    Tokens.
    str: Output clean data from duplicated lettre.



* **Return type**

    list



### sudoai.preprocess.strip_punc(text: str)
Clean text from punctuation.


* **Parameters**

    **text** (*str*) – Input data text.



* **Returns**

    Output text cleaned from punctuation.



* **Return type**

    str



### sudoai.preprocess.unique_chars(path: str, do_lower_case: bool = False)
Get unique characters from data file.


* **Parameters**

    
    * **path** (*str*) – Path of text data.


    * **do_lower_case** (*bool**, **optional*) – If True lower case text. Defaults to True.



* **Raises**

    **FileExistsError** – When data file not exist.



* **Returns**

    Unique characters.



* **Return type**

    set(str)



### sudoai.preprocess.unique_words(path: str, clean: bool = False, do_lower_case: bool = True, strip_duplicated: bool = False, stopword: Optional[sudoai.preprocess.text.StopWord] = None)
Get unique words from text file.


* **Parameters**

    
    * **path** (*str*) – Path of text data.


    * **clean** (*bool**, **optional*) – If True clean text. Defaults to False.


    * **do_lower_case** (*bool**, **optional*) – If True lower case text. Defaults to True.


    * **strip_duplicated** (*bool**, **optional*) – If True clean text from duplicated characters. Defaults to False.


    * **stopword** (`StopWord`, optional) – StopWord class to escape stopwords . Defaults to None.



* **Raises**

    **FileExistsError** – When text file not exist.



* **Returns**

    Unique words.



* **Return type**

    set



### sudoai.preprocess.unique_words_with_pattern(path, pattern='[a-z0-9]+', do_lower_case=True, clean=False, strip_duplicated=False, stopword=None)

### sudoai.preprocess.whitespace_tokenize(text: str)
Tokenizer text with spaces.


* **Parameters**

    **text** (*str*) – Input data text.



* **Returns**

    Output tokens.



* **Return type**

    list



### sudoai.preprocess.word_frequency(path: str, word_freq: Optional[collections.Counter] = None, do_lower_case: bool = True, clean: bool = False, strip_duplicated: bool = False, stopword: Optional[sudoai.preprocess.text.StopWord] = None)
Get word frequency from text file.


* **Parameters**

    
    * **path** (*str*) – Path of text data.


    * **word_freq** (`Counter`, optional) – Old Counter to update it with new Counter. Defaults to None.


    * **do_lower_case** (*bool**, **optional*) – If True lower case text. Defaults to True.


    * **clean** (*bool**, **optional*) – If True clean text. Defaults to False.


    * **strip_duplicated** (*bool**, **optional*) – If True clean text from duplicated characters. Defaults to False.


    * **stopword** (`StopWord`, optional) – StopWord class to escape stopwords . Defaults to None.



* **Raises**

    **FileExistsError** – When data file not exist.



* **Returns**

    Words frequency.



* **Return type**

    `Counter`
