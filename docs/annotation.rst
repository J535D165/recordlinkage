**********
Annotation
**********

Manually labeled record pairs are useful in training and validation tasks.
Training data is usually not available in record linkage applications because
it is highly dataset and sample-specific. The Python Record Linkage Toolkit
comes with a `browser-based user interface`_ for manually classifying record
pairs. A hosted version of `RecordLinkage ANNOTATOR`_ can be found on Github.

.. _`browser-based user interface`: https://github.com/J535D165/recordlinkage-annotator
.. _`RecordLinkage ANNOTATOR`: https://j535d165.github.io/recordlinkage-annotator/

.. image:: https://github.com/J535D165/recordlinkage-annotator/blob/master/images/annotator_review.png?raw=true
   :alt: Review screen of RecordLinkage ANNOTATOR
   :target: https://j535d165.github.io/recordlinkage-annotator/

Generate annotation file
========================

The `RecordLinkage ANNOTATOR`_ software requires a structured annotation
file. The required schema_ of the annotation file is open. The function
:func:`recordlinkage.write_annotation_file` can be used to render and save an
annotation file. The function can be used for both linking and deduplication
purposes. 

.. _schema: https://github.com/J535D165/recordlinkage-annotator/tree/master/schema

.. autofunction:: recordlinkage.write_annotation_file

Linking
-------

This is a simple example of the code to render an annotation
file for linking records:

.. code:: python

    import recordlinkage as rl
    from recordlinkage.index import Block
    from recordlinkage.datasets import load_febrl4

    df_a, df_b = load_febrl4()

    blocker = Block("surname", "surname")
    pairs = blocker.index(df_a, df_b)

    rl.write_annotation_file(
        "annotation_demo_linking.json",
        pairs[0:50],
        df_a,
        df_b,
        dataset_a_name="Febrl4 A",
        dataset_b_name="Febrl4 B"
    )

Deduplication
-------------

This is a simple example of the code to render an annotation
file for duplicate detection:

.. code:: python

    import recordlinkage as rl
    from recordlinkage.index import Block
    from recordlinkage.datasets import load_febrl1

    df_a = load_febrl1()

    blocker = Block("surname", "surname")
    pairs = blocker.index(df_a)

    rl.write_annotation_file(
        "annotation_demo_dedup.json",
        pairs[0:50],
        df_a,
        dataset_a_name="Febrl1 A"
    )


Manual labeling
===============

Go to `RecordLinkage ANNOTATOR`_ or start the server yourself. 

Choose the annotation file on the landing screen or use the drag and drop
functionality. A new screen shows the first record pair to label. Start
labeling data the manually. Use the button `Match` for record pairs belonging
to the same entity. Use `Distinct` for record pairs belonging to different
entities. After all records are labeled by hand, the result can be saved to a
file.


Export/read annotation file
===========================

After labeling all record pairs, you can export the annotation file to a JSON
file. Use the function :func:`recordlinkage.read_annotation_file` to read the
results. 

.. code:: python

    import recordlinkage as rl

    result = rl.read_annotation_file('my_annotation.json')
    print(result.links)

The function :func:`recordlinkage.read_annotation_file` reads the file and returns 
an :class:`recordlinkage.annotation.AnnotationResult` object. This object contains 
links and distinct attributes that return a :class:`pandas.MultiIndex` object.

.. autofunction:: recordlinkage.read_annotation_file


.. autoclass:: recordlinkage.annotation.AnnotationResult 
    :members:
    :inherited-members:

