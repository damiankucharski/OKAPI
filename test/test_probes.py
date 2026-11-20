from itertools import product

import numpy as np
import pytest

from okapi.probes import first_probe_declaration, probe_log_evalues


def test_evidence_has_unit_mean_under_fair_randomization():
    assignments = np.asarray(list(product((0.,1.), repeat=9)))[...,None]
    evidence = probe_log_evalues(assignments,1-assignments)
    np.testing.assert_allclose(np.exp(evidence).mean(axis=0),1.,rtol=1e-12)
    decision = first_probe_declaration(evidence)
    assert np.mean(decision.index>=0) <= .05


def test_fixed_and_joint_crossings_and_no_effect():
    active=np.zeros((3,32,2))
    active[0,:,0]=1
    active[1,:,:]=1
    evidence=probe_log_evalues(active,np.zeros_like(active))
    decision=first_probe_declaration(evidence)
    assert list(decision.index)==[0,-1,-1]
    assert 1<decision.records[0]<32 and decision.records[0]==decision.records[1]
    assert decision.records[2]==32
    with pytest.raises(ValueError):
        probe_log_evalues(active+2,active)
