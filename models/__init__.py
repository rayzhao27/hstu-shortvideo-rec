"""Stages 2 and 3: feed Stage 1 KuaiRand sequences into Meta's HSTU, then train it.

    python -m utils.meta_repo              # clone the official repo into third_party/
    python -m models.smoke                 # Stage 2: one batch, forward + backward
    python -m models.config                # the size ladder
    python -m models.train --smoke-test    # Stage 3: whole loop, tiny, asserts resume
    python -m models.train --size base --out-dir runs/base
    python -m models.evaluate --checkpoint runs/base/checkpoints/best.pt --split test

Which file owns what:

* ``dataset.py``  Stage 1 ``*_seqs.pkl`` -> padded tensors, plus truncation coverage.
* ``hstu.py``     the official HSTU encoder, the item/action interleave, and the loss.
* ``config.py``   named model shapes; Stage 5's scaling law reuses them by name.
* ``evaluate.py`` full-catalogue Recall@K / NDCG@K per protocol stream, with a
                  popularity baseline and the chance rate as reference points.
* ``train.py``    the loop: TensorBoard, periodic and best checkpoints, auto-resume.
"""
