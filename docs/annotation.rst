**********
Annotation
**********

Labeled record pairs can be useful for training and validation classification
models. Training data is usually not available in record linkage applications
because it is highly dataset and sample-specific. The Python Record Linkage
Toolkit comes with a browser-based user interface for manually classifying
record pairs. A hosted version of `RecordLinkage ANNOTATOR`_ can be found on
Github.

.. _`RecordLinkage ANNOTATOR`: https://j535d165.github.io/recordlinkage-annotator/

.. image:: https://github.com/J535D165/recordlinkage-annotator/blob/master/images/annotator_review.png?raw=true
   :alt: Review screen of RecordLinkage ANNOTATOR
   :target: https://j535d165.github.io/recordlinkage-annotator/

Generate annotation file
========================

The function `recordlinkage.write_annotator_file()` renders and saves an
annotation file. The function can be used for both linking and deduplication
purposes. This is a simple example of the code needed to render an annotation
file:

.. code:: python

    import recordlinkage as rl
    from recordlinkage.index import Block
    from recordlinkage.datasets import load_febrl4

    df_a, df_b = load_febrl4()

    blocker = Block("surname", "surname")
    pairs = blocker.index(df_a, df_b)

    rl.write_annotation_file(
        "annotation_demo.json",
        pairs[0:50],
        df_a,
        df_b,
        dataset_a_name="Febrl4 A",
        dataset_b_name="Febrl4 B"
    )


.. autofunction:: recordlinkage.write_annotation_file


Manual labeling
===============

Go to `RecordLinkage ANNOTATOR`_ or start the server yourself. 

Load the annotation file on the landing screen and start labeling data
manually. Use the button `Match` for record pairs belonging to the same
entity. Use `Distinct` for record pairs belonging to different entities.

Save the result to a file. 


Export/read annotation file
===========================

After labeling all record pairs, you can export the annotation file to a JSON
file. Use the function `recordlinkage.read_annotation_file` to read the
results. 

.. code:: python

    import recordlinkage as rl

    result = rl.read_annotation_file('my_annotation.json')
    print(result.links)


.. autofunction:: recordlinkage.read_annotation_file


.. autoclass:: recordlinkage.annotation.AnnotationResult 
    :members:
    :inherited-members:

