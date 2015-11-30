
.. code:: python

    %cd -q ..

.. code:: python

    import pandas
    import numpy
    
    import recordlinkage
    from recordlinkage import datasets

.. code:: python

    dfA = datasets.load_censusA()
    dfB = datasets.load_censusB()
    
    print "The number of record in dataset A: %s" % len(dfA)
    print "The number of record in dataset B: %s" % len(dfB)
    
    dfB.head()


.. parsed-literal::

    The number of record in dataset A: 1000
    The number of record in dataset B: 1000




.. raw:: html

    <div>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>birthdate</th>
          <th>city</th>
          <th>email</th>
          <th>entity_id</th>
          <th>first_name</th>
          <th>job</th>
          <th>last_name</th>
          <th>phone_number</th>
          <th>postcode</th>
          <th>sex</th>
          <th>street_address</th>
        </tr>
        <tr>
          <th>record_id</th>
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
          <th>1000000</th>
          <td>1970-05-17</td>
          <td>New Alphons</td>
          <td>bahringer.kaia@hotmail.com</td>
          <td>77</td>
          <td>Mindy</td>
          <td>Engineer, broadcasting (operations)</td>
          <td>NaN</td>
          <td>(216)197-1192</td>
          <td>65643-0681</td>
          <td>F</td>
          <td>9705 Coletta Crossing Apt. 109</td>
        </tr>
        <tr>
          <th>1000001</th>
          <td>1989-06-27</td>
          <td>South Shavon</td>
          <td>stoltenberg.mazie@hotmail.com</td>
          <td>662</td>
          <td>Shawna</td>
          <td>IT technical support officer</td>
          <td>Blanda</td>
          <td>637.010.1504</td>
          <td>94928-6143</td>
          <td>F</td>
          <td>9098 Dorathea Knoll</td>
        </tr>
        <tr>
          <th>1000002</th>
          <td>1972-06-16</td>
          <td>North Elinoreshire</td>
          <td>lyn.damore@yahoo.com</td>
          <td>638</td>
          <td>Alexa</td>
          <td>Armed forces training and education officer</td>
          <td>Lebsack</td>
          <td>123.170.2312x9612</td>
          <td>87228-7861</td>
          <td>F</td>
          <td>6681 Hessel River</td>
        </tr>
        <tr>
          <th>1000003</th>
          <td>1989-05-25</td>
          <td>Lake Elbertaland</td>
          <td>elinor.hane@hotmail.com</td>
          <td>207</td>
          <td>Linnie</td>
          <td>Civil engineer, contracting</td>
          <td>O'Hara</td>
          <td>NaN</td>
          <td>57146</td>
          <td>M</td>
          <td>9252 Lesley Mountain</td>
        </tr>
        <tr>
          <th>1000004</th>
          <td>1994-10-18</td>
          <td>Kuvalismouth</td>
          <td>joelle63@gmail.com</td>
          <td>295</td>
          <td>Wava</td>
          <td>Teacher, special educational needs</td>
          <td>Cassin</td>
          <td>06871533331</td>
          <td>09199-7265</td>
          <td>F</td>
          <td>0122 Kadin Flat Apt. 785</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: python

    std_dfA = recordlinkage.StandardDataFrame(dfA)

Create an object Pairs to create candidate record pairs. The class
'Pairs' can take 1 or 2 arguments, both dataframes. If one dataframe is
given, the record pairs are for deduplication. When two dataframes are
given, the data is linked between two files.

.. code:: python

    pairing = recordlinkage.Pairs(dfA, dfB, suffixes=('_dfA', '_dfB'))

The simplest method of comparing record pairs is to compare all possible
records pairs. The method 'full' is used for this. The function takes no
arguments.

.. code:: python

    pairs_full = pairing.full()
    print "The reduction ratio is %s" % pairing.reduction_ratio()
    pairs_full.head(10)


.. parsed-literal::

    The reduction ratio is 0.0




.. raw:: html

    <div>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th></th>
          <th>birthdate_dfA</th>
          <th>city_dfA</th>
          <th>email_dfA</th>
          <th>first_name_dfA</th>
          <th>job_dfA</th>
          <th>last_name_dfA</th>
          <th>phone_number_dfA</th>
          <th>postcode_dfA</th>
          <th>sex_dfA</th>
          <th>street_address_dfA</th>
          <th>...</th>
          <th>city_dfB</th>
          <th>email_dfB</th>
          <th>entity_id_dfB</th>
          <th>first_name_dfB</th>
          <th>job_dfB</th>
          <th>last_name_dfB</th>
          <th>phone_number_dfB</th>
          <th>postcode_dfB</th>
          <th>sex_dfB</th>
          <th>street_address_dfB</th>
        </tr>
        <tr>
          <th>index_dfA</th>
          <th>index_dfB</th>
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
          <th rowspan="10" valign="top">1000000</th>
          <th>1000000</th>
          <td>1975-05-18</td>
          <td>Champlinville</td>
          <td>hakeem.vonrueden@gmail.com</td>
          <td>Marchello</td>
          <td>Private music teacher</td>
          <td>Prosacco</td>
          <td>1-002-603-2082x8411</td>
          <td>06419-6160</td>
          <td>M</td>
          <td>4937 Jerimy Knolls</td>
          <td>...</td>
          <td>New Alphons</td>
          <td>bahringer.kaia@hotmail.com</td>
          <td>77</td>
          <td>Mindy</td>
          <td>Engineer, broadcasting (operations)</td>
          <td>NaN</td>
          <td>(216)197-1192</td>
          <td>65643-0681</td>
          <td>F</td>
          <td>9705 Coletta Crossing Apt. 109</td>
        </tr>
        <tr>
          <th>1000001</th>
          <td>1975-05-18</td>
          <td>Champlinville</td>
          <td>hakeem.vonrueden@gmail.com</td>
          <td>Marchello</td>
          <td>Private music teacher</td>
          <td>Prosacco</td>
          <td>1-002-603-2082x8411</td>
          <td>06419-6160</td>
          <td>M</td>
          <td>4937 Jerimy Knolls</td>
          <td>...</td>
          <td>South Shavon</td>
          <td>stoltenberg.mazie@hotmail.com</td>
          <td>662</td>
          <td>Shawna</td>
          <td>IT technical support officer</td>
          <td>Blanda</td>
          <td>637.010.1504</td>
          <td>94928-6143</td>
          <td>F</td>
          <td>9098 Dorathea Knoll</td>
        </tr>
        <tr>
          <th>1000002</th>
          <td>1975-05-18</td>
          <td>Champlinville</td>
          <td>hakeem.vonrueden@gmail.com</td>
          <td>Marchello</td>
          <td>Private music teacher</td>
          <td>Prosacco</td>
          <td>1-002-603-2082x8411</td>
          <td>06419-6160</td>
          <td>M</td>
          <td>4937 Jerimy Knolls</td>
          <td>...</td>
          <td>North Elinoreshire</td>
          <td>lyn.damore@yahoo.com</td>
          <td>638</td>
          <td>Alexa</td>
          <td>Armed forces training and education officer</td>
          <td>Lebsack</td>
          <td>123.170.2312x9612</td>
          <td>87228-7861</td>
          <td>F</td>
          <td>6681 Hessel River</td>
        </tr>
        <tr>
          <th>1000003</th>
          <td>1975-05-18</td>
          <td>Champlinville</td>
          <td>hakeem.vonrueden@gmail.com</td>
          <td>Marchello</td>
          <td>Private music teacher</td>
          <td>Prosacco</td>
          <td>1-002-603-2082x8411</td>
          <td>06419-6160</td>
          <td>M</td>
          <td>4937 Jerimy Knolls</td>
          <td>...</td>
          <td>Lake Elbertaland</td>
          <td>elinor.hane@hotmail.com</td>
          <td>207</td>
          <td>Linnie</td>
          <td>Civil engineer, contracting</td>
          <td>O'Hara</td>
          <td>NaN</td>
          <td>57146</td>
          <td>M</td>
          <td>9252 Lesley Mountain</td>
        </tr>
        <tr>
          <th>1000004</th>
          <td>1975-05-18</td>
          <td>Champlinville</td>
          <td>hakeem.vonrueden@gmail.com</td>
          <td>Marchello</td>
          <td>Private music teacher</td>
          <td>Prosacco</td>
          <td>1-002-603-2082x8411</td>
          <td>06419-6160</td>
          <td>M</td>
          <td>4937 Jerimy Knolls</td>
          <td>...</td>
          <td>Kuvalismouth</td>
          <td>joelle63@gmail.com</td>
          <td>295</td>
          <td>Wava</td>
          <td>Teacher, special educational needs</td>
          <td>Cassin</td>
          <td>06871533331</td>
          <td>09199-7265</td>
          <td>F</td>
          <td>0122 Kadin Flat Apt. 785</td>
        </tr>
        <tr>
          <th>1000005</th>
          <td>1975-05-18</td>
          <td>Champlinville</td>
          <td>hakeem.vonrueden@gmail.com</td>
          <td>Marchello</td>
          <td>Private music teacher</td>
          <td>Prosacco</td>
          <td>1-002-603-2082x8411</td>
          <td>06419-6160</td>
          <td>M</td>
          <td>4937 Jerimy Knolls</td>
          <td>...</td>
          <td>Port Kandi</td>
          <td>cleon11@gmail.com</td>
          <td>838</td>
          <td>Jerrold</td>
          <td>Estate manager/land agent</td>
          <td>Mraz</td>
          <td>1-952-440-9167x108</td>
          <td>NaN</td>
          <td>M</td>
          <td>829 Tonja Mission Suite 329</td>
        </tr>
        <tr>
          <th>1000006</th>
          <td>1975-05-18</td>
          <td>Champlinville</td>
          <td>hakeem.vonrueden@gmail.com</td>
          <td>Marchello</td>
          <td>Private music teacher</td>
          <td>Prosacco</td>
          <td>1-002-603-2082x8411</td>
          <td>06419-6160</td>
          <td>M</td>
          <td>4937 Jerimy Knolls</td>
          <td>...</td>
          <td>Weimannshire</td>
          <td>kerluke.catherine@yahoo.com</td>
          <td>345</td>
          <td>Permelia</td>
          <td>Health and safety inspector</td>
          <td>Wolff</td>
          <td>(220)786-1831</td>
          <td>86541</td>
          <td>F</td>
          <td>226 Aylin Extension</td>
        </tr>
        <tr>
          <th>1000007</th>
          <td>1975-05-18</td>
          <td>Champlinville</td>
          <td>hakeem.vonrueden@gmail.com</td>
          <td>Marchello</td>
          <td>Private music teacher</td>
          <td>Prosacco</td>
          <td>1-002-603-2082x8411</td>
          <td>06419-6160</td>
          <td>M</td>
          <td>4937 Jerimy Knolls</td>
          <td>...</td>
          <td>Hegmannbury</td>
          <td>muller.shellie@gmail.com</td>
          <td>215</td>
          <td>Claiborne</td>
          <td>NaN</td>
          <td>Bayer</td>
          <td>572.091.8319x9248</td>
          <td>98292-2375</td>
          <td>M</td>
          <td>0109 Alia Avenue Suite 501</td>
        </tr>
        <tr>
          <th>1000008</th>
          <td>1975-05-18</td>
          <td>Champlinville</td>
          <td>hakeem.vonrueden@gmail.com</td>
          <td>Marchello</td>
          <td>Private music teacher</td>
          <td>Prosacco</td>
          <td>1-002-603-2082x8411</td>
          <td>06419-6160</td>
          <td>M</td>
          <td>4937 Jerimy Knolls</td>
          <td>...</td>
          <td>South Pearlineberg</td>
          <td>arch93@yahoo.com</td>
          <td>309</td>
          <td>Billie</td>
          <td>Local government officer</td>
          <td>Mann</td>
          <td>713.657.8963x91709</td>
          <td>15155</td>
          <td>F</td>
          <td>056 Altenwerth Curve</td>
        </tr>
        <tr>
          <th>1000009</th>
          <td>1975-05-18</td>
          <td>Champlinville</td>
          <td>hakeem.vonrueden@gmail.com</td>
          <td>Marchello</td>
          <td>Private music teacher</td>
          <td>Prosacco</td>
          <td>1-002-603-2082x8411</td>
          <td>06419-6160</td>
          <td>M</td>
          <td>4937 Jerimy Knolls</td>
          <td>...</td>
          <td>Elainemouth</td>
          <td>kuhlman.cleo@gmail.com</td>
          <td>446</td>
          <td>Sampson</td>
          <td>Conservation officer, nature</td>
          <td>Trantow</td>
          <td>1-784-370-7883x3350</td>
          <td>11299-7671</td>
          <td>M</td>
          <td>905 Caddie Overpass Suite 932</td>
        </tr>
      </tbody>
    </table>
    <p>10 rows × 22 columns</p>
    </div>



Make pairs based on a blocking key. This blocking key can be a list of
blocking keys. In this case 'Surname' was used.

.. code:: python

    pairs_block = pairing.block('last_name')
    print "The reduction ratio is %s" % pairing.reduction_ratio()
    
    pairs_block.head(10)


.. parsed-literal::

    The reduction ratio is 0.997108




.. raw:: html

    <div>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th></th>
          <th>birthdate_dfA</th>
          <th>city_dfA</th>
          <th>email_dfA</th>
          <th>first_name_dfA</th>
          <th>job_dfA</th>
          <th>last_name</th>
          <th>phone_number_dfA</th>
          <th>postcode_dfA</th>
          <th>sex_dfA</th>
          <th>street_address_dfA</th>
          <th>...</th>
          <th>birthdate_dfB</th>
          <th>city_dfB</th>
          <th>email_dfB</th>
          <th>entity_id_dfB</th>
          <th>first_name_dfB</th>
          <th>job_dfB</th>
          <th>phone_number_dfB</th>
          <th>postcode_dfB</th>
          <th>sex_dfB</th>
          <th>street_address_dfB</th>
        </tr>
        <tr>
          <th>index_dfA</th>
          <th>index_dfB</th>
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
          <th rowspan="2" valign="top">1000000</th>
          <th>1000349</th>
          <td>1975-05-18</td>
          <td>Champlinville</td>
          <td>hakeem.vonrueden@gmail.com</td>
          <td>Marchello</td>
          <td>Private music teacher</td>
          <td>Prosacco</td>
          <td>1-002-603-2082x8411</td>
          <td>06419-6160</td>
          <td>M</td>
          <td>4937 Jerimy Knolls</td>
          <td>...</td>
          <td>1975-05-18</td>
          <td>Champlinville</td>
          <td>hakeem.vonrueden@gmail.com</td>
          <td>1</td>
          <td>Marchello</td>
          <td>Private music teacher</td>
          <td>1-002-603-2082x8411</td>
          <td>06419-6160</td>
          <td>M</td>
          <td>4937 Jerimy Knolls</td>
        </tr>
        <tr>
          <th>1000868</th>
          <td>1975-05-18</td>
          <td>Champlinville</td>
          <td>hakeem.vonrueden@gmail.com</td>
          <td>Marchello</td>
          <td>Private music teacher</td>
          <td>Prosacco</td>
          <td>1-002-603-2082x8411</td>
          <td>06419-6160</td>
          <td>M</td>
          <td>4937 Jerimy Knolls</td>
          <td>...</td>
          <td>1996-12-18</td>
          <td>East Macktown</td>
          <td>chynna.stanton@gmail.com</td>
          <td>NaN</td>
          <td>Doctor</td>
          <td>Lobbyist</td>
          <td>NaN</td>
          <td>75525</td>
          <td>M</td>
          <td>59390 Dedric Summit</td>
        </tr>
        <tr>
          <th rowspan="2" valign="top">1000727</th>
          <th>1000349</th>
          <td>1991-11-16</td>
          <td>Lake Webbton</td>
          <td>channie47@hotmail.com</td>
          <td>Harlow</td>
          <td>Retail manager</td>
          <td>Prosacco</td>
          <td>713.168.2785</td>
          <td>72901-7555</td>
          <td>M</td>
          <td>642 Schmidt Pike</td>
          <td>...</td>
          <td>1975-05-18</td>
          <td>Champlinville</td>
          <td>hakeem.vonrueden@gmail.com</td>
          <td>1</td>
          <td>Marchello</td>
          <td>Private music teacher</td>
          <td>1-002-603-2082x8411</td>
          <td>06419-6160</td>
          <td>M</td>
          <td>4937 Jerimy Knolls</td>
        </tr>
        <tr>
          <th>1000868</th>
          <td>1991-11-16</td>
          <td>Lake Webbton</td>
          <td>channie47@hotmail.com</td>
          <td>Harlow</td>
          <td>Retail manager</td>
          <td>Prosacco</td>
          <td>713.168.2785</td>
          <td>72901-7555</td>
          <td>M</td>
          <td>642 Schmidt Pike</td>
          <td>...</td>
          <td>1996-12-18</td>
          <td>East Macktown</td>
          <td>chynna.stanton@gmail.com</td>
          <td>NaN</td>
          <td>Doctor</td>
          <td>Lobbyist</td>
          <td>NaN</td>
          <td>75525</td>
          <td>M</td>
          <td>59390 Dedric Summit</td>
        </tr>
        <tr>
          <th rowspan="2" valign="top">1000821</th>
          <th>1000349</th>
          <td>1970-06-05</td>
          <td>Lake Trudie</td>
          <td>gerlach.javonte@gmail.com</td>
          <td>Odelia</td>
          <td>Human resources officer</td>
          <td>Prosacco</td>
          <td>(702)196-7724</td>
          <td>97026</td>
          <td>F</td>
          <td>5913 Crist Wells Suite 335</td>
          <td>...</td>
          <td>1975-05-18</td>
          <td>Champlinville</td>
          <td>hakeem.vonrueden@gmail.com</td>
          <td>1</td>
          <td>Marchello</td>
          <td>Private music teacher</td>
          <td>1-002-603-2082x8411</td>
          <td>06419-6160</td>
          <td>M</td>
          <td>4937 Jerimy Knolls</td>
        </tr>
        <tr>
          <th>1000868</th>
          <td>1970-06-05</td>
          <td>Lake Trudie</td>
          <td>gerlach.javonte@gmail.com</td>
          <td>Odelia</td>
          <td>Human resources officer</td>
          <td>Prosacco</td>
          <td>(702)196-7724</td>
          <td>97026</td>
          <td>F</td>
          <td>5913 Crist Wells Suite 335</td>
          <td>...</td>
          <td>1996-12-18</td>
          <td>East Macktown</td>
          <td>chynna.stanton@gmail.com</td>
          <td>NaN</td>
          <td>Doctor</td>
          <td>Lobbyist</td>
          <td>NaN</td>
          <td>75525</td>
          <td>M</td>
          <td>59390 Dedric Summit</td>
        </tr>
        <tr>
          <th rowspan="4" valign="top">1000001</th>
          <th>1000339</th>
          <td>2000-11-15</td>
          <td>Jewelview</td>
          <td>koch.aditya@gmail.com</td>
          <td>Linna</td>
          <td>Trade union research officer</td>
          <td>Dietrich</td>
          <td>1-300-313-9491</td>
          <td>09014-2947</td>
          <td>F</td>
          <td>93722 Hermina Stream Apt. 244</td>
          <td>...</td>
          <td>1997-09-14</td>
          <td>NaN</td>
          <td>velda.mclaughlin@yahoo.com</td>
          <td>270</td>
          <td>Ferris</td>
          <td>Ambulance person</td>
          <td>NaN</td>
          <td>66416</td>
          <td>M</td>
          <td>0282 Ankunding Highway Apt. 537</td>
        </tr>
        <tr>
          <th>1000534</th>
          <td>2000-11-15</td>
          <td>Jewelview</td>
          <td>koch.aditya@gmail.com</td>
          <td>Linna</td>
          <td>Trade union research officer</td>
          <td>Dietrich</td>
          <td>1-300-313-9491</td>
          <td>09014-2947</td>
          <td>F</td>
          <td>93722 Hermina Stream Apt. 244</td>
          <td>...</td>
          <td>1990-10-25</td>
          <td>New Shaunna</td>
          <td>imanol.jones@gmail.com</td>
          <td>502</td>
          <td>Tuan</td>
          <td>Health and safety inspector</td>
          <td>845-043-5524x475</td>
          <td>49482</td>
          <td>M</td>
          <td>28468 Tiera Knolls Apt. 598</td>
        </tr>
        <tr>
          <th>1000560</th>
          <td>2000-11-15</td>
          <td>Jewelview</td>
          <td>koch.aditya@gmail.com</td>
          <td>Linna</td>
          <td>Trade union research officer</td>
          <td>Dietrich</td>
          <td>1-300-313-9491</td>
          <td>09014-2947</td>
          <td>F</td>
          <td>93722 Hermina Stream Apt. 244</td>
          <td>...</td>
          <td>1984-06-21</td>
          <td>Lucinabury</td>
          <td>xhagenes@hotmail.com</td>
          <td>978</td>
          <td>Dillie</td>
          <td>Passenger transport manager</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>F</td>
          <td>23681 Dorthey Springs Apt. 675</td>
        </tr>
        <tr>
          <th>1000849</th>
          <td>2000-11-15</td>
          <td>Jewelview</td>
          <td>koch.aditya@gmail.com</td>
          <td>Linna</td>
          <td>Trade union research officer</td>
          <td>Dietrich</td>
          <td>1-300-313-9491</td>
          <td>09014-2947</td>
          <td>F</td>
          <td>93722 Hermina Stream Apt. 244</td>
          <td>...</td>
          <td>1983-12-31</td>
          <td>Lake Floybury</td>
          <td>rashad91@yahoo.com</td>
          <td>NaN</td>
          <td>Levon</td>
          <td>Health visitor</td>
          <td>NaN</td>
          <td>49246</td>
          <td>M</td>
          <td>1270 Lana Flats Suite 842</td>
        </tr>
      </tbody>
    </table>
    <p>10 rows × 21 columns</p>
    </div>



.. code:: python

    pairs_sorted = pairing.sortedneighbourhood('last_name', window=3)
    print "The reduction ratio is %s" % pairing.reduction_ratio()
    
    pairs_sorted.head(10)


.. parsed-literal::

    The reduction ratio is 0.983523




.. raw:: html

    <div>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th></th>
          <th>email_dfA</th>
          <th>sex_dfA</th>
          <th>sex_dfB</th>
          <th>phone_number_dfB</th>
          <th>street_address_dfA</th>
          <th>email_dfB</th>
          <th>city_dfA</th>
          <th>birthdate_dfA</th>
          <th>last_name_dfA</th>
          <th>last_name_dfB</th>
          <th>...</th>
          <th>postcode_dfA</th>
          <th>postcode_dfB</th>
          <th>job_dfA</th>
          <th>job_dfB</th>
          <th>city_dfB</th>
          <th>phone_number_dfA</th>
          <th>entity_id_dfB</th>
          <th>street_address_dfB</th>
          <th>first_name_dfA</th>
          <th>first_name_dfB</th>
        </tr>
        <tr>
          <th>index_dfA</th>
          <th>index_dfB</th>
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
          <th rowspan="3" valign="top">1000000</th>
          <th>1000051</th>
          <td>hakeem.vonrueden@gmail.com</td>
          <td>M</td>
          <td>F</td>
          <td>668.380.9142</td>
          <td>4937 Jerimy Knolls</td>
          <td>taya47@gmail.com</td>
          <td>Champlinville</td>
          <td>1975-05-18</td>
          <td>Prosacco</td>
          <td>Quitzon</td>
          <td>...</td>
          <td>06419-6160</td>
          <td>46270-5131</td>
          <td>Private music teacher</td>
          <td>Physiological scientist</td>
          <td>Pacochachester</td>
          <td>1-002-603-2082x8411</td>
          <td>571</td>
          <td>NaN</td>
          <td>Marchello</td>
          <td>Cordie</td>
        </tr>
        <tr>
          <th>1000152</th>
          <td>hakeem.vonrueden@gmail.com</td>
          <td>M</td>
          <td>M</td>
          <td>201.528.2199x580</td>
          <td>4937 Jerimy Knolls</td>
          <td>NaN</td>
          <td>Champlinville</td>
          <td>1975-05-18</td>
          <td>Prosacco</td>
          <td>Quitzon</td>
          <td>...</td>
          <td>06419-6160</td>
          <td>00078</td>
          <td>Private music teacher</td>
          <td>Financial planner</td>
          <td>West Mannie</td>
          <td>1-002-603-2082x8411</td>
          <td>678</td>
          <td>75848 Balistreri Mission</td>
          <td>Marchello</td>
          <td>Jeramy</td>
        </tr>
        <tr>
          <th>1000755</th>
          <td>hakeem.vonrueden@gmail.com</td>
          <td>M</td>
          <td>F</td>
          <td>882-056-2000</td>
          <td>4937 Jerimy Knolls</td>
          <td>alena52@gmail.com</td>
          <td>Champlinville</td>
          <td>1975-05-18</td>
          <td>Prosacco</td>
          <td>Quitzon</td>
          <td>...</td>
          <td>06419-6160</td>
          <td>62675-7170</td>
          <td>Private music teacher</td>
          <td>Designer, blown glass/stained glass</td>
          <td>North Phylisland</td>
          <td>1-002-603-2082x8411</td>
          <td>775</td>
          <td>2057 Logan Wells</td>
          <td>Marchello</td>
          <td>Chanelle</td>
        </tr>
        <tr>
          <th rowspan="3" valign="top">1000727</th>
          <th>1000051</th>
          <td>channie47@hotmail.com</td>
          <td>M</td>
          <td>F</td>
          <td>668.380.9142</td>
          <td>642 Schmidt Pike</td>
          <td>taya47@gmail.com</td>
          <td>Lake Webbton</td>
          <td>1991-11-16</td>
          <td>Prosacco</td>
          <td>Quitzon</td>
          <td>...</td>
          <td>72901-7555</td>
          <td>46270-5131</td>
          <td>Retail manager</td>
          <td>Physiological scientist</td>
          <td>Pacochachester</td>
          <td>713.168.2785</td>
          <td>571</td>
          <td>NaN</td>
          <td>Harlow</td>
          <td>Cordie</td>
        </tr>
        <tr>
          <th>1000152</th>
          <td>channie47@hotmail.com</td>
          <td>M</td>
          <td>M</td>
          <td>201.528.2199x580</td>
          <td>642 Schmidt Pike</td>
          <td>NaN</td>
          <td>Lake Webbton</td>
          <td>1991-11-16</td>
          <td>Prosacco</td>
          <td>Quitzon</td>
          <td>...</td>
          <td>72901-7555</td>
          <td>00078</td>
          <td>Retail manager</td>
          <td>Financial planner</td>
          <td>West Mannie</td>
          <td>713.168.2785</td>
          <td>678</td>
          <td>75848 Balistreri Mission</td>
          <td>Harlow</td>
          <td>Jeramy</td>
        </tr>
        <tr>
          <th>1000755</th>
          <td>channie47@hotmail.com</td>
          <td>M</td>
          <td>F</td>
          <td>882-056-2000</td>
          <td>642 Schmidt Pike</td>
          <td>alena52@gmail.com</td>
          <td>Lake Webbton</td>
          <td>1991-11-16</td>
          <td>Prosacco</td>
          <td>Quitzon</td>
          <td>...</td>
          <td>72901-7555</td>
          <td>62675-7170</td>
          <td>Retail manager</td>
          <td>Designer, blown glass/stained glass</td>
          <td>North Phylisland</td>
          <td>713.168.2785</td>
          <td>775</td>
          <td>2057 Logan Wells</td>
          <td>Harlow</td>
          <td>Chanelle</td>
        </tr>
        <tr>
          <th rowspan="3" valign="top">1000821</th>
          <th>1000051</th>
          <td>gerlach.javonte@gmail.com</td>
          <td>F</td>
          <td>F</td>
          <td>668.380.9142</td>
          <td>5913 Crist Wells Suite 335</td>
          <td>taya47@gmail.com</td>
          <td>Lake Trudie</td>
          <td>1970-06-05</td>
          <td>Prosacco</td>
          <td>Quitzon</td>
          <td>...</td>
          <td>97026</td>
          <td>46270-5131</td>
          <td>Human resources officer</td>
          <td>Physiological scientist</td>
          <td>Pacochachester</td>
          <td>(702)196-7724</td>
          <td>571</td>
          <td>NaN</td>
          <td>Odelia</td>
          <td>Cordie</td>
        </tr>
        <tr>
          <th>1000152</th>
          <td>gerlach.javonte@gmail.com</td>
          <td>F</td>
          <td>M</td>
          <td>201.528.2199x580</td>
          <td>5913 Crist Wells Suite 335</td>
          <td>NaN</td>
          <td>Lake Trudie</td>
          <td>1970-06-05</td>
          <td>Prosacco</td>
          <td>Quitzon</td>
          <td>...</td>
          <td>97026</td>
          <td>00078</td>
          <td>Human resources officer</td>
          <td>Financial planner</td>
          <td>West Mannie</td>
          <td>(702)196-7724</td>
          <td>678</td>
          <td>75848 Balistreri Mission</td>
          <td>Odelia</td>
          <td>Jeramy</td>
        </tr>
        <tr>
          <th>1000755</th>
          <td>gerlach.javonte@gmail.com</td>
          <td>F</td>
          <td>F</td>
          <td>882-056-2000</td>
          <td>5913 Crist Wells Suite 335</td>
          <td>alena52@gmail.com</td>
          <td>Lake Trudie</td>
          <td>1970-06-05</td>
          <td>Prosacco</td>
          <td>Quitzon</td>
          <td>...</td>
          <td>97026</td>
          <td>62675-7170</td>
          <td>Human resources officer</td>
          <td>Designer, blown glass/stained glass</td>
          <td>North Phylisland</td>
          <td>(702)196-7724</td>
          <td>775</td>
          <td>2057 Logan Wells</td>
          <td>Odelia</td>
          <td>Chanelle</td>
        </tr>
        <tr>
          <th>1000001</th>
          <th>1000070</th>
          <td>koch.aditya@gmail.com</td>
          <td>F</td>
          <td>F</td>
          <td>242-467-9418</td>
          <td>93722 Hermina Stream Apt. 244</td>
          <td>NaN</td>
          <td>Jewelview</td>
          <td>2000-11-15</td>
          <td>Dietrich</td>
          <td>Doyle</td>
          <td>...</td>
          <td>09014-2947</td>
          <td>61606</td>
          <td>Trade union research officer</td>
          <td>Purchasing manager</td>
          <td>Yostburgh</td>
          <td>1-300-313-9491</td>
          <td>863</td>
          <td>878 Marquardt Point</td>
          <td>Linna</td>
          <td>Tara</td>
        </tr>
      </tbody>
    </table>
    <p>10 rows × 22 columns</p>
    </div>



.. code:: python

    pairs_sorted_with_block = pairing.sortedneighbourhood('last_name', window=3, blocking_on=['sex'])
    pairs_sorted_with_block.head(10)





.. raw:: html

    <div>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th></th>
          <th>job_dfA</th>
          <th>email_dfB</th>
          <th>postcode_dfA</th>
          <th>email_dfA</th>
          <th>first_name_dfB</th>
          <th>city_dfA</th>
          <th>street_address_dfB</th>
          <th>sex</th>
          <th>first_name_dfA</th>
          <th>entity_id_dfA</th>
          <th>...</th>
          <th>city_dfB</th>
          <th>birthdate_dfA</th>
          <th>last_name_dfA</th>
          <th>last_name_dfB</th>
          <th>birthdate_dfB</th>
          <th>postcode_dfB</th>
          <th>street_address_dfA</th>
          <th>phone_number_dfA</th>
          <th>entity_id_dfB</th>
          <th>phone_number_dfB</th>
        </tr>
        <tr>
          <th>index_dfA</th>
          <th>index_dfB</th>
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
          <th>1000000</th>
          <th>1000152</th>
          <td>Private music teacher</td>
          <td>NaN</td>
          <td>06419-6160</td>
          <td>hakeem.vonrueden@gmail.com</td>
          <td>Jeramy</td>
          <td>Champlinville</td>
          <td>75848 Balistreri Mission</td>
          <td>M</td>
          <td>Marchello</td>
          <td>1</td>
          <td>...</td>
          <td>West Mannie</td>
          <td>1975-05-18</td>
          <td>Prosacco</td>
          <td>Quitzon</td>
          <td>1977-05-19</td>
          <td>00078</td>
          <td>4937 Jerimy Knolls</td>
          <td>1-002-603-2082x8411</td>
          <td>678</td>
          <td>201.528.2199x580</td>
        </tr>
        <tr>
          <th>1000727</th>
          <th>1000152</th>
          <td>Retail manager</td>
          <td>NaN</td>
          <td>72901-7555</td>
          <td>channie47@hotmail.com</td>
          <td>Jeramy</td>
          <td>Lake Webbton</td>
          <td>75848 Balistreri Mission</td>
          <td>M</td>
          <td>Harlow</td>
          <td>728</td>
          <td>...</td>
          <td>West Mannie</td>
          <td>1991-11-16</td>
          <td>Prosacco</td>
          <td>Quitzon</td>
          <td>1977-05-19</td>
          <td>00078</td>
          <td>642 Schmidt Pike</td>
          <td>713.168.2785</td>
          <td>678</td>
          <td>201.528.2199x580</td>
        </tr>
        <tr>
          <th rowspan="2" valign="top">1000001</th>
          <th>1000070</th>
          <td>Trade union research officer</td>
          <td>NaN</td>
          <td>09014-2947</td>
          <td>koch.aditya@gmail.com</td>
          <td>Tara</td>
          <td>Jewelview</td>
          <td>878 Marquardt Point</td>
          <td>F</td>
          <td>Linna</td>
          <td>2</td>
          <td>...</td>
          <td>Yostburgh</td>
          <td>2000-11-15</td>
          <td>Dietrich</td>
          <td>Doyle</td>
          <td>1983-02-21</td>
          <td>61606</td>
          <td>93722 Hermina Stream Apt. 244</td>
          <td>1-300-313-9491</td>
          <td>863</td>
          <td>242-467-9418</td>
        </tr>
        <tr>
          <th>1000906</th>
          <td>Trade union research officer</td>
          <td>hkessler@hotmail.com</td>
          <td>09014-2947</td>
          <td>koch.aditya@gmail.com</td>
          <td>Blanca</td>
          <td>Jewelview</td>
          <td>078 Carry Centers</td>
          <td>F</td>
          <td>Linna</td>
          <td>2</td>
          <td>...</td>
          <td>West Frona</td>
          <td>2000-11-15</td>
          <td>Dietrich</td>
          <td>Doyle</td>
          <td>2001-02-27</td>
          <td>10140-1665</td>
          <td>93722 Hermina Stream Apt. 244</td>
          <td>1-300-313-9491</td>
          <td>NaN</td>
          <td>(984)946-0892</td>
        </tr>
        <tr>
          <th rowspan="2" valign="top">1000977</th>
          <th>1000070</th>
          <td>Passenger transport manager</td>
          <td>NaN</td>
          <td>25065</td>
          <td>xhagenes@hotmail.com</td>
          <td>Tara</td>
          <td>Lucinabury</td>
          <td>878 Marquardt Point</td>
          <td>F</td>
          <td>Dillie</td>
          <td>978</td>
          <td>...</td>
          <td>Yostburgh</td>
          <td>1984-06-21</td>
          <td>Dietrich</td>
          <td>Doyle</td>
          <td>1983-02-21</td>
          <td>61606</td>
          <td>23681 Dorthey Springs Apt. 675</td>
          <td>1-480-367-9913x67284</td>
          <td>863</td>
          <td>242-467-9418</td>
        </tr>
        <tr>
          <th>1000906</th>
          <td>Passenger transport manager</td>
          <td>hkessler@hotmail.com</td>
          <td>25065</td>
          <td>xhagenes@hotmail.com</td>
          <td>Blanca</td>
          <td>Lucinabury</td>
          <td>078 Carry Centers</td>
          <td>F</td>
          <td>Dillie</td>
          <td>978</td>
          <td>...</td>
          <td>West Frona</td>
          <td>1984-06-21</td>
          <td>Dietrich</td>
          <td>Doyle</td>
          <td>2001-02-27</td>
          <td>10140-1665</td>
          <td>23681 Dorthey Springs Apt. 675</td>
          <td>1-480-367-9913x67284</td>
          <td>NaN</td>
          <td>(984)946-0892</td>
        </tr>
        <tr>
          <th>1000005</th>
          <th>1000009</th>
          <td>Trade mark attorney</td>
          <td>kuhlman.cleo@gmail.com</td>
          <td>53642-6501</td>
          <td>strosin.mal@gmail.com</td>
          <td>Sampson</td>
          <td>Lake Dennisfort</td>
          <td>905 Caddie Overpass Suite 932</td>
          <td>M</td>
          <td>Franklin</td>
          <td>6</td>
          <td>...</td>
          <td>Elainemouth</td>
          <td>2002-01-18</td>
          <td>Torphy</td>
          <td>Trantow</td>
          <td>1987-04-05</td>
          <td>11299-7671</td>
          <td>668 Brakus Lock Apt. 958</td>
          <td>205.960.9156</td>
          <td>446</td>
          <td>1-784-370-7883x3350</td>
        </tr>
        <tr>
          <th>1000581</th>
          <th>1000009</th>
          <td>Broadcast engineer</td>
          <td>kuhlman.cleo@gmail.com</td>
          <td>31074</td>
          <td>dianne.konopelski@hotmail.com</td>
          <td>Sampson</td>
          <td>West Brittny</td>
          <td>905 Caddie Overpass Suite 932</td>
          <td>M</td>
          <td>Percy</td>
          <td>582</td>
          <td>...</td>
          <td>Elainemouth</td>
          <td>1973-02-26</td>
          <td>Torphy</td>
          <td>Trantow</td>
          <td>1987-04-05</td>
          <td>11299-7671</td>
          <td>6709 Herman Forks</td>
          <td>1-456-699-7884x2500</td>
          <td>446</td>
          <td>1-784-370-7883x3350</td>
        </tr>
        <tr>
          <th>1000008</th>
          <th>1000007</th>
          <td>Advertising art director</td>
          <td>muller.shellie@gmail.com</td>
          <td>79308-8130</td>
          <td>raegan.roberts@hotmail.com</td>
          <td>Claiborne</td>
          <td>Port Medoraview</td>
          <td>0109 Alia Avenue Suite 501</td>
          <td>M</td>
          <td>Marshal</td>
          <td>9</td>
          <td>...</td>
          <td>Hegmannbury</td>
          <td>1976-11-14</td>
          <td>Batz</td>
          <td>Bayer</td>
          <td>2001-04-16</td>
          <td>98292-2375</td>
          <td>03091 Dwaine Falls Apt. 707</td>
          <td>1-103-191-6024</td>
          <td>215</td>
          <td>572.091.8319x9248</td>
        </tr>
        <tr>
          <th>1000009</th>
          <th>1000413</th>
          <td>Oceanographer</td>
          <td>xthiel@gmail.com</td>
          <td>67625-2721</td>
          <td>vonrueden.benedict@hotmail.com</td>
          <td>Corinne</td>
          <td>Bednarmouth</td>
          <td>4559 Cami Rapids</td>
          <td>F</td>
          <td>Margretta</td>
          <td>10</td>
          <td>...</td>
          <td>East Kalie</td>
          <td>1982-06-11</td>
          <td>Hansen</td>
          <td>Harvey</td>
          <td>2011-10-10</td>
          <td>98725</td>
          <td>095 Schimmel Corner</td>
          <td>006-521-5252x3785</td>
          <td>659</td>
          <td>NaN</td>
        </tr>
      </tbody>
    </table>
    <p>10 rows × 21 columns</p>
    </div>



.. code:: python

    for pairs in pairing.iterindex(recordlinkage.indexing._fullindex, 1000,1000):
        
        if not pairs.empty:
            print 'block'

This is the same as the following code.

.. code:: python

    for pairs in pairing.iterfull(1000,1000):
        
        pass

Also blocking and sorted neighbourhood indexing can be used with
iterations. The number of records pairs is not always equal for each
iteration.

.. code:: python

    for pairs in pairing.iterblock(1000,1000, ' surname'):
    
        pass

.. code:: python

    comp = recordlinkage.Compare(pairs_block)
    
    # print pairs_block.columns
    comp.compare(recordlinkage.comparing.exact, 
                 pairs_block[[' given_name_dfA',' address_2_dfA']], 
                 pairs_block[[' given_name_dfB', ' address_2_dfB']], 
                 missing_value=9, 
                 output='any',
                 name='given_name')
    comp.compare(recordlinkage.comparing.exact, 
                 pairs_block[' given_name_dfA'], 
                 pairs_block[' address_2_dfB'], 
                 missing_value=9, 
                 output='any',
                 name='sur_name')
    # comp.compare(comparing.exact, pairs_block[' address_2_dfA'], pairs_block[' address_2_dfB'], missing_value=np.nan , name='address')
    
    # pairs_block.loc[pairs_block.ix[[0,1,2,3]].index,' given_name_dfA'] = 3
    # print sum(pairs_block[' given_name_dfB'].isnull())
    
    comp.comparison_vectors

.. code:: python

    fs = recordlinkage.FellegiSunterClassifier()
    
    fs.estimate(
        comparison_vectors=comp.comparison_vectors, 
        start_m={'given_name':{0:0.1, 1:0.7, 9:0.2},'sur_name':{0:0.1, 9:0.9}},
        start_u={'given_name':{0:0.7, 1:0.1, 9: 0.2},'sur_name':{0:0.9, 9:0.1}},
        start_p=0.1,
        max_iter=10
    )
    
    print fs.est.number_of_pairs
    print fs.est.p
    fs.est.summary()
    
    len(fs.auto_classify(comp.comparison_vectors))
    
    
    


.. code:: python

    x = pd.Series([1,2,3,4])
    y = pd.Series([1,2,3,4])
    
    x.name= 'test'
    y.name = 'red'
    print x 
    print y
    
    pd.concat([x,y], axis=1)

.. code:: python

    %matplotlib inline
    import matplotlib.pyplot as plt
    
    import networkx as nx
    
    B = nx.Graph()
    B.add_nodes_from([1,2], bipartite=0) # Add the node attribute "bipartite"
    B.add_nodes_from([3], bipartite=1)
    B.add_edges_from([(1,1), (1,2)])
    
    nx.draw(B)

.. code:: python

    reload(recordlinkage)
    import itertools
    
    test_data = recordlinkage.StandardSeries(dfA[' given_name'].copy())

.. code:: python

    test_data.group_similar_values()

.. code:: python

    %timeit list(itertools.combinations(test_data[test_data.notnull()].astype(unicode).unique(), 2))
