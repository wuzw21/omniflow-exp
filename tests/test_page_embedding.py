from __future__ import annotations

import numpy as np

from omniflow.transfer.embedding import PageEncoder


def test_page_encoder_uses_one_canonical_schema_for_learned_512d_words() -> None:
    encoder = PageEncoder()
    parse_count = 0
    graph_from_record = encoder._schema._graph_from_record

    def count_parse(*args, **kwargs):
        nonlocal parse_count
        parse_count += 1
        return graph_from_record(*args, **kwargs)

    encoder._schema._graph_from_record = count_parse

    page = encoder.embed(
        '<hierarchy><node package="com.example" text="Search" '
        'class="android.widget.EditText" bounds="[0,0][100,40]"/></hierarchy>'
    )

    assert encoder.name == "page_vector"
    assert encoder.version.startswith("page-vector.v3:")
    assert encoder.element_dimension == 64
    assert encoder.dimension == 512
    assert encoder.checkpoint_sha256 == (
        "c262f03c32c4b88d2933323fe2b33007281224ef1a8aae1418a9844d354de232"
    )
    assert page.vector.shape == (512,)
    assert len(page.elements) == 1
    assert page.elements[0].id == "0.0"
    assert page.elements[0].parent_id == "0"
    assert parse_count == 1
    assert np.isclose(np.linalg.norm(page.vector), 1.0)
