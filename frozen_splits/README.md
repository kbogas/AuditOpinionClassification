This folder will contain the data used in this study.

Because of the nature of the data, we can't share the AuditAnalytic labels for each opinion.

Please download the zip file, containing the dataset split already accordinly.

The data are stored in *.jsonl* format like the following: 

```cmd
[
   {
      "Source Date (Year)": "20XX",
      "Company": "Company Name",
      "OPINION_TEXT": "Full report from SEC",
      "OPINION_SUMMARY": "Summary generated from the report"
      "id": 1234
   },
   ...

```

The fields shown correspond to:

- *Source Date (Year)*: The year of the report
- *Company*: The company name
- *OPINION_TEXT*: The original text of the report from SEC
- *OPINION_SUMMARY*: A summary version of the report (created by us for the *summary* variant)
- *id*: An integer used as a unique id for this report