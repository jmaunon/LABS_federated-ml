## BigQuery Federated Analytics
 
The goal of the experiment is to perform a Exploratory Data Analysis (EDA) with [the churn for bank customers dataset](https://www.kaggle.com/adammaus/predicting-churn-for-bank-customers). 
 
This EDA is developed without moving the external stored data where the queries are executed. Concretely, this analytics process is executed remotely in BigQuery.

For this purpose, 2 silos of data are used:
 - 1 data silo in GCS
 - 1 data silo in CloudSQL

In order to execute the notebook, it is required to setup the federated analytics environment. This [document](https://docs.google.com/document/d/1SadrpgQN6aUTSYZYc4wisvOKaugZZVay1VekigFx4EU/edit) describes the process.

The final conclusions of the experiment could be found at the end of this [presentation](https://docs.google.com/presentation/d/1FUS9MYljOwSsdea1T-t6VnfCzDR1U5Q-HQNxva_ZQz8/edit#slide=id.g10964f20bf8_0_103).
