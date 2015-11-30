
The following example shows how the record linkage module can be used to
link two files with personal information. First import the Recordlinkage
module and the Pandas module. The Recordlinkage module is build on the
strong Pandas framework.

.. code:: python

    %cd -q ..

.. code:: python

    import pandas
    import recordlinkage
    
    from recordlinkage import datasets

After importing the record linkage and pandas module, it is time to
import the datasets. In this case, we use two sample datasets with basic
information about persons. Call the dataset dfA and dfB.

.. code:: python

    dfA = datasets.load_censusA()
    dfB = datasets.load_censusB()

The first step is to make record pairs. Each pair contains one record of
dataset dfA and one record of dataset dfB. The pairs contain the
information of all columns in dfA and dfB.

.. code:: python

    pairing = recordlinkage.Pairs(dfA, dfB, suffixes=('_A', '_B'))
    
    pairs = pairing.full()

Each record pair is compared on some attributes both record have in
common.

.. code:: python

    comparing = recordlinkage.Compare(pairs)
    comparing.exact(pairs['first_name_A'], pairs['first_name_B'], name='first_name')
    comparing.exact(pairs['last_name_A'], pairs['last_name_B'], name='last_name')
    comparing.exact(pairs['sex_A'], pairs['sex_B'], name='sex')
    comparing.exact(pairs['birthdate_A'], pairs['birthdate_B'], name='birthdate')
    comparing.exact(pairs['city_A'], pairs['city_B'], name='city')
    comparing.exact(pairs['street_address_A'], pairs['street_address_B'], name='street_address')
    comparing.exact(pairs['job_A'], pairs['job_B'], name='job')
    comparing.exact(pairs['email_A'], pairs['email_B'], name='email')
    
    comparing.comparison_vectors.sample(10)




.. raw:: html

    <div>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th></th>
          <th>first_name</th>
          <th>last_name</th>
          <th>sex</th>
          <th>birthdate</th>
          <th>city</th>
          <th>street_address</th>
          <th>job</th>
          <th>email</th>
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
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>1000101</th>
          <th>1000182</th>
          <td>0</td>
          <td>0</td>
          <td>1</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
        </tr>
        <tr>
          <th>1000500</th>
          <th>1000536</th>
          <td>0</td>
          <td>0</td>
          <td>1</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
        </tr>
        <tr>
          <th>1000273</th>
          <th>1000148</th>
          <td>0</td>
          <td>0</td>
          <td>1</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
        </tr>
        <tr>
          <th>1000651</th>
          <th>1000706</th>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
        </tr>
        <tr>
          <th>1000742</th>
          <th>1000366</th>
          <td>0</td>
          <td>0</td>
          <td>1</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
        </tr>
        <tr>
          <th>1000870</th>
          <th>1000502</th>
          <td>0</td>
          <td>0</td>
          <td>1</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
        </tr>
        <tr>
          <th>1000246</th>
          <th>1000472</th>
          <td>0</td>
          <td>0</td>
          <td>1</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
        </tr>
        <tr>
          <th>1000164</th>
          <th>1000104</th>
          <td>0</td>
          <td>1</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
        </tr>
        <tr>
          <th>1000194</th>
          <th>1000887</th>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
        </tr>
        <tr>
          <th>1000543</th>
          <th>1000872</th>
          <td>0</td>
          <td>0</td>
          <td>0</td>
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
