# Pretraining of language models

This repository implement pretraining of Huggingface language models using MLM. This repository was created as part of a project for [Digital Revisor](https://www.digitalrevisor.nu/) which wanted to rework their ML pipeline to accomodate other languages than english. This repository was used to create a [ELECTRA](https://arxiv.org/abs/2003.10555) model for [Dutch](https://huggingface.co/ajders/nl_electra).

All parameters for training, models and datasets are set in `src/config.py`.

Developers:

- Anders Jess Pedersen (anders.j.pedersen@alexandra.dk)
