Quickstart
===============

Introduction
^^^^^^^^^^^^^^

This is where you describe sudoai framework.

Install
^^^^^^^^^^^^^^

SUDOAI is private repo, To use it you must have ssh key.

1) Install SUDOAI from github::

    pip install git+ssh://git@github.com/suai-tn/sudo-ai.git

2) Clone repo and install SUDOAI::

    pip clone git@github.com:suai-tn/sudo-ai.git
    cd sudo-ai
    pip install .

Transliteration with sudoai
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With sudoai we can create a lot of NLP project.
In this quickstart we try to create transliteration solution
form arabizi to Tunisian Arabic Dialect.


Before beginning
-----------------

Check the current version.

>>> import sudoai
>>> sudoai.__version__
0.1.2

You can use sudoai log.

>>> sudoai.__log__.info('help')
[INFO -- 07-25 00:33:17 ] -- help

Data files
-----------------

We have three data files (src.txt, t.txt and ttd.txt).

'src.txt' file contain source words::

    maandich
    mahlol
    manaref
    fdhaye7

't.txt' file contain target words::

    معنديش
    محلول
    منعرف
    فضايح

'ttd.txt' file contain source and target with ',' as separator::

    monia,منية
    raj3ouni,رجعوني
    ytawle,يطول



Tokenizers
-----------------

Tokenization is the process of breaking down a piece of text into small units called tokens.
A token may be a word, part of a word or just characters.
In our case token is a character, so we use :obj:`sudoai.preprocess.CharTokenizer` for the both side
source and target. 

Create source and target Tokenizers::

    from sudoai.preprocess import CharTokenizer

    t_src = CharTokenizer(strip_duplicated=True, do_lower_case=True)
    t_src.train(path="src.txt")
    t_src.save('ttd/src', True)

    t_target = CharTokenizer(strip_duplicated=True, do_lower_case=True)
    t_target.train(path="t.txt")
    t_target.save('ttd/target', True)


Dataset
-----------------

With sudoai you should create dataset to samplify the training process.
Remember 'ttd.txt' file that contain source and target with ',' as separator,
we gona use it to create our dataset::

    from sudoai.dataset import Dataset , DatasetInfo , DataType , DatasetType
    from sudoai.utils import ZipAlgo

    info = DatasetInfo(id='ttd',
                        version='0.1.0',
                        data_path='ttd.txt',
                        sep=',',
                        data_type=DataType.TEXT,
                        dataset_type=DatasetType.WORD_TO_WORD)
    dataset = Dataset(info=info, src_tokenizer=t_src, target_tokenizer=t_target)
    dataset.build()

    dataset.save(override=True, is_ziped=True, algo=ZipAlgo.LZMA)

Our dataset createdsxxwsx and saved with id 'ttd', now we can create auto hypertuning to check the best
params for our model.

Hypertuning
-----------------

Auto hypertuning based on ax platform, with different params to check the the best combinaison possible.
You should have a dataset saved for hypertuning module.
Create hypertuning for the model word to word::

    from sudoai.hypertuning import w2w_hypertuning

    w2w_hypertuning(dataset_id='ttd',n_experience=200)

After w2w_hypertuning() finish we have the best params, so we train our model and save the checkpoints.
Example of w2w_hypertuning() result::

    ({'learning_rate': 5.258774659629995e-05,
    'momentum': 0.47246561050415037,
    'hidden_size': 512,
    'loss': 'crossentropy',
    'optimizer': 'adam'},
    ({'acc': 0.37416824866086246,
    'val_acc': 0.4478062566369772,
    'loss': 2.4569015627620807,
    'val_loss': 2.0418924935670155},
    {'acc': {'acc': nan},
    'val_acc': {'val_acc': nan},
    'loss': {'loss': nan},
    'val_loss': {'val_loss': nan}}))


Trainer
-----------------

To train word to word model with 100 epochs::

    from sudoai.trainer import Word2WordTrainer 

    w2w = Word2WordTrainer(
        id='ttd',
        hidden_size=512,
        lr=0.00001,
        momentum=0.47,
        loss='crossentropy',
        optimizer='adam',
        do_eval=True,
        do_shuffle=True,
        epochs=100,
        do_save=True,
        do_save_checkpoint=True,
        base_path='my/base/path/',
        wandb_key='mywandbkey'
    )

    w2w()


Pipeline
----------------

To use pipeline first you should create pipeline config file::

    from sudoai.pipeline import PipelineConfig, save_pipeline_config, ModelType

    pipe_config = PipelineConfig(id='ttd', is_two_tokenizer=True, model_type = ModelType.WORD_TO_WORD)
    save_pipeline_config(config=pipe_config, override=True)

Load and use pipeline ::

    from sudoai.pipeline import Pipeline

    nlp = Pipeline(id='ttd', version='0.1.0')
    _output = nlp(inputs='chbik')
    print(_output)

    'شبيك'

With this sample lines of code you can create a strong NLP project with sudoai.
For more information check all the docs, have a nice code.

