Settings configurations used for different models.


Here 3 are given as examples pertaining to *Longformer*, *DistilBert (summary)* and *DistilBert weighted*.

These are passed to the fine-tuning script as argument. For example, if we wanted to train and evaluate a Longformer model we would run (from the parent folder):

```cmd
python fine_tune_lm.py ./settings/longformer_unfrozen
```