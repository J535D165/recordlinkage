
Basic example
=============


.. code:: python

    import pandas
    import recordlinkage

After importing the record linkage and pandas module, it is time to
import the datasets. In this case, we use two sample datasets with basic
information about persons. Call the dataset dfA and dfB.

.. code:: python

    dfA = pandas.read_csv('sampledata/dataset2.csv', index_col='rec_id')
    dfB = pandas.read_csv('sampledata/dataset1.csv', index_col='rec_id')

The first step is to make record pairs. Each pair contains one record of
dataset dfA and one record of dataset dfB. The pairs contain the
information of all columns in dfA and dfB.

.. code:: python

    pairing = recordlinkage.Pairs(dfA, dfB)
    
    pairs = pairing.full()

Each record pair is compared on some attributes both record have in
common.

.. code:: python

    comparing = recordlinkage.Compare(pairs)
    comparing.compare(recordlinkage.exact, pairs['surname_A'], pairs['surname_B'], name='surname')
    comparing.compare(recordlinkage.exact, pairs['given_name_A'], pairs['given_name_B'], name='given_name')
    comparing.compare(recordlinkage.exact, pairs['suburb_A'], pairs['suburb_B'], name='suburb')
    comparing.compare(recordlinkage.exact, pairs['date_of_birth_A'], pairs['date_of_birth_B'], name='date_of_birth')
    comparing.compare(recordlinkage.exact, pairs['street_number_A'], pairs['street_number_B'], name='street_number')
    
    comparing.comparison_vectors.head()




.. raw:: html

    <div>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th></th>
          <th>surname</th>
          <th>given_name</th>
          <th>suburb</th>
          <th>date_of_birth</th>
          <th>street_number</th>
        </tr>
        <tr>
          <th>index_A</th>
          <th>index_B</th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th rowspan="5" valign="top">rec-2778-org</th>
          <th>rec-223-org</th>
          <td>0</td>
          <td>NaN</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
        </tr>
        <tr>
          <th>rec-122-org</th>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
        </tr>
        <tr>
          <th>rec-373-org</th>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
        </tr>
        <tr>
          <th>rec-10-dup-0</th>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>rec-227-org</th>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: python

    m = {
        'surname': {0:0.1, 1:0.9},
        'given_name': {0:0.1, 1:0.9},
        'suburb': {0:0.1, 1:0.9},
        'date_of_birth': {0:0.1, 1:0.9},
        'street_number': {0:0.1, 1:0.9}
        }
    
    u = {
        'surname': {0:0.9, 1:0.1},
        'given_name': {0:0.9, 1:0.1},
        'suburb': {0:0.9, 1:0.1},
        'date_of_birth': {0:0.9, 1:0.1},
        'street_number': {0:0.9, 1:0.1}
        }
    
    p = 0.1
    
    fs = recordlinkage.FellegiSunterClassifier()
    
    fs.ecm(
        comparison_vectors=comparing.comparison_vectors, 
        start_m=m, 
        start_u=u, 
        start_p=p,
        max_iter=20
    )
    
    fs.matches(comparing.comparison_vectors, w=10)




.. raw:: html

    <div>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th></th>
          <th>surname</th>
          <th>given_name</th>
          <th>suburb</th>
          <th>date_of_birth</th>
          <th>street_number</th>
          <th>count</th>
          <th>m</th>
          <th>u</th>
          <th>weight</th>
          <th>p_M</th>
          <th>p</th>
          <th>lambda</th>
          <th>mu</th>
        </tr>
        <tr>
          <th>index_A</th>
          <th>index_B</th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>rec-2151-dup-3</th>
          <th>rec-172-org</th>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>1</td>
          <td>1</td>
          <td>5</td>
          <td>0.121</td>
          <td>3.358949e-07</td>
          <td>12.794506</td>
          <td>0.715494</td>
          <td>0.000007</td>
          <td>0.993637</td>
          <td>3.383640e-07</td>
        </tr>
        <tr>
          <th>rec-2151-dup-4</th>
          <th>rec-172-org</th>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>1</td>
          <td>1</td>
          <td>5</td>
          <td>0.121</td>
          <td>3.358949e-07</td>
          <td>12.794506</td>
          <td>0.715494</td>
          <td>0.000007</td>
          <td>0.993637</td>
          <td>3.383640e-07</td>
        </tr>
        <tr>
          <th>rec-2151-dup-2</th>
          <th>rec-172-org</th>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>1</td>
          <td>1</td>
          <td>5</td>
          <td>0.121</td>
          <td>3.358949e-07</td>
          <td>12.794506</td>
          <td>0.715494</td>
          <td>0.000007</td>
          <td>0.993637</td>
          <td>3.383640e-07</td>
        </tr>
        <tr>
          <th>rec-2151-org</th>
          <th>rec-172-org</th>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>1</td>
          <td>1</td>
          <td>5</td>
          <td>0.121</td>
          <td>3.358949e-07</td>
          <td>12.794506</td>
          <td>0.715494</td>
          <td>0.000007</td>
          <td>0.993637</td>
          <td>3.383640e-07</td>
        </tr>
        <tr>
          <th>rec-2151-dup-0</th>
          <th>rec-172-org</th>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>1</td>
          <td>1</td>
          <td>5</td>
          <td>0.121</td>
          <td>3.358949e-07</td>
          <td>12.794506</td>
          <td>0.715494</td>
          <td>0.000007</td>
          <td>0.993637</td>
          <td>3.383640e-07</td>
        </tr>
      </tbody>
    </table>
    </div>


