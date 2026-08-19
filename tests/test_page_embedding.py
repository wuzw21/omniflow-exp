from __future__ import annotations

import numpy as np

from omniflow.transfer.page_embedding import OmniTransferPageEncoder


def test_page_encoder_uses_canonical_unified_contextual_checkpoint() -> None:
    encoder = OmniTransferPageEncoder(device="cpu")

    page = encoder.embed(
        '<hierarchy><node text="Search" class="android.widget.EditText" '
        'bounds="[0,0][100,40]"/></hierarchy>'
    )

    assert encoder.checkpoint_path.name == "relation_slots_l3_h64_seed17.npz"
    assert encoder.checkpoint_sha256 == (
        "c262f03c32c4b88d2933323fe2b33007281224ef1a8aae1418a9844d354de232"
    )
    assert encoder.dimension == 64
    assert page.element_count == 2
    assert "numpy-unified-association-v1" in page.encoder_version
    assert np.isclose(np.linalg.norm(page.vector), 1.0)
