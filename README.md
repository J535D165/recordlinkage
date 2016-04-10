# recordlinkage

This *recordlinkage* package is a library to link records in or between data sources. The package provides most of the tools needed for record linkage. The package contains indexing methods, functions to compare records and classifiers. The package is developed for research and linking of small or medium sized files. 

This project is inspired by the **Freely Extensible Biomedical Record Linkage (FEBRL)** project, which is a great project. This project has one big difference, it uses ``pandas`` and ``numpy`` for data handling and computations. The use of ``pandas``, a flexible and powerful data analysis and manipulation library for Python,  makes the record linkage process much easier. A lot of built in ``pandas`` methods can be used to integrate your record linkage directly into existing data manipulation projects.  

One of the aims of this project is to make an extensible record linkage framework. It is easy to include your own indexing algorithms, comparison/similarity measures and classifiers. 

## Simple linking example
Import the ``recordlinkage`` module that contains all important tools for the linking of records. Also import two **pandas** dataframes with example data named ``censusdataA`` and ``censusdataB``. 

```python
import recordlinkage
import pandas
```
Consider that we have two datasets with personal information. Load these datasets into a pandas dataframe.
```python 
dfA = pandas.DataFrame(YOUR_FIRST_DATASET)
dfB = pandas.DataFrame(YOUR_SECOND_DATASET)
```

Next, we are going to decide which record pairs are interesting to evaluate fully. This is done by making a ``pandas.MultiIndex``.

```python
index = recordlinkage.Index(dfA, dfB)
candidate_links = index.block('surname')
```

For each record pair, we compare the records from both dataframes.
```python
compare = recordlinkage.Compare(candidate_links, dfA, dfB)

compare.fuzzy('name', 'name', method='jarowinkler', threshold=0.85)
compare.exact('sex', 'gender')
compare.exact('dob', 'date_of_birth')
compare.fuzzy('streetname', 'streetname', method='damerau_levenshtein', threshold=0.7)
compare.exact('place', 'placename')
compare.exact('haircolor', 'haircolor', missing_value=9)
```

At each point, you can get the comparison vectors by the following attribute:
```python
compare.vectors
```

```python
recordlinkage.FellegiSunter()
```



## Main Features
The main features of the **recordlinkage** package are:

  - Clean and standardise data
  - Make pairs of records with several indexing methods such as **blocking** and **sorted neighbourhood indexing**
  - Compare records with a large number of comparison and similarity functions (for example: the jaro-winkler and levenshtein distance)
  - Several classifications algorithms, both supervised and unsupervised algorithms. 

## Dependencies, installation and license
The following packages are required. You probably have it already ;)
- [numpy](http://www.numpy.org): 1.7.0 or higher
- [pandas](https://github.com/pydata/pandas): 0.17.0 or higher
- [sklearn](http://scikit-learn.org/)

The following packages are recommanded
- [jellyfish](https://github.com/jamesturk/jellyfish): Needed for approximate string comparison. Version 0.5.0 or higher.

It is not possible to install the package with ``pip``. You can download or clone the **recordlinkage** project and install it in the normal way

```sh
python setup.py install
```
The license for this record linkage tool is GPLv3.
