Commands
=========

Commande line tools for sudoai.

sudoai-io
^^^^^^^^^^^^^^^^^^

sudoai-io Commande line to Download and Upload pipelines, dataset and models.

Parameters 
-------------

    | --upload (bool, optional): To upload pipeline, dataset and model. Defaults to False.
    | --download (bool, optional): To download pipeline, dataset and model. Defaults to True. 
    | --id (str): Unique ID of pipeline, dataset or model.

Raises 
----------
    | TypeError: When you set --upload and --download to True.
    | TypeError: When you don't set --upload and --download.

Examples ::

    sudoai-io --download=True --id=trans
    sudoai-io --upload=True --id=ttd
    sudoai-io --id=ttd

sudoai-ch2m
^^^^^^^^^^^^^^^^^^

sudoai-ch2m Commande line to generate model from checkpoint.

Parameters 
-------------

    | --chp (str): Path of checkpoint.
    | --id (str): Unique ID of model.
    | --mtype (str, optional): Model Type. Defaults to w2w.
    | --train (bool, optional): Train mode. Defaults to False.
    | --version (str, optional): Model version. Defaults to '0.1.0'.

Examples ::

    sudoai-ch2m --chp=checkpoint-1.chp --id=trans --train=False
    sudoai-ch2m --chp=checkpoint-2.chp --id=trans
    sudoai-ch2m --chp=checkpoint-1.chp --id=trans --train=False --version=0.1.0 --mtype=w2w

