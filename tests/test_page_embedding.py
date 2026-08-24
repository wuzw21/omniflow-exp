from __future__ import annotations

import numpy as np

from omniflow.transfer.embedding import PageEncoder


def test_page_encoder_uses_canonical_v10_learned_state_attention() -> None:
    encoder = PageEncoder()
    page = encoder.embed(
        '<hierarchy><node package="com.example" text="Search" '
        'class="android.widget.EditText" bounds="[0,0][100,40]"/></hierarchy>'
    )

    assert encoder.name == "omnitransfer_state_attention"
    assert encoder.version == "omnitransfer-v10:learned-state-attention"
    assert encoder.element_dimension == 0
    assert encoder.dimension == 1024
    assert encoder.architecture == "omnitransfer_point_conditioned_sparse_graph_v10"
    assert encoder.checkpoint_sha256 == (
        "3b783ed113fc37397e2f092d133e970ec36bdbf0d26e262d9273389e4729d16f"
    )
    assert page.vector.shape == (1024,)
    assert page.node_count == 2
    assert page.elements == ()
    assert np.isclose(np.linalg.norm(page.vector), 1.0)
