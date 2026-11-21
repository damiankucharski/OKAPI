# Two indistinguishable sensor pairings with different target labels

Freeze this design before computing experimental scores. Eight sensors lie
in a uniformly relabeled alternating cycle. Alternating edges form two
disjoint perfect matchings A and H. Draw an independent fair sign per
matched pair in each family and per example. Sensor j reports positive-class
score sigmoid(1.2 × (A_g + H_h + e_j)), where the two signs correspond to
its incident pairs, and e_j is independent Gaussian noise with standard
deviation 0.60 on the source. Share the entire prediction matrix between
two worlds. World A's label is the OR of two ANDs on its four positive
signs, and world H uses the same rule on its own four signs. Independently
randomize the assignment of pairs to paths in each world and bank.

The marginal sensor law is identical across worlds. Under the equal world
prior, a procedure using predictions alone can choose the correct single
matching in at most one world per bank. In the population, logit correlations
are positive exactly on the cycle edges. A single even cycle has exactly
two perfect matchings composed only of these edges. The experiment asks
whether both are among the two highest-scoring proposals at finite fit size,
and whether ordinary labeled tree scoring resolves the remaining ambiguity.

## Methods and splits

Use 12 independent banks, seeds 41200–41211. Each world fits on 384 rows,
validates on 256 separately drawn rows, then scores on 4,096 IID rows and
4,096 independently drawn shifted rows with noise standard deviation 1.25.
Provider IDs are plain positions after the hidden random relabeling. The
two worlds train independently using their own labels and the same
optimizer seed. Freeze both worlds' models before drawing test examples.

Rank all 105 perfect matchings by the sum of their four fit-time Pearson
correlations of logit scores. Break exact ties lexicographically. Over each
proposed matching, enumerate all 120 commutatively distinct read-once
binary min/max trees on the four within-pair probability means, retaining
functionally equivalent trees. Compare the highest-scoring matching alone
(120 fit assessments), the two highest (240), and two uniform distinct
random matchings (240). Each policy validates the four candidates with
lowest fit Brier, using stable ties, and takes the lowest validation Brier.
Also run stock-operator GP at 240 fit assessments with an independent
four-finalist validation, arithmetic mean, and a logit logistic stacker
with fixed C=1 using the same 384 fit labels. The two simple baselines use
zero labels for selection. Candidate scoring uses a single world's binary
Brier. Proposal construction receives only its prediction matrix. No test
labels or latent assignments enter matching, tree fitting, or validation.

As a structural reference, apply the exact matching and path arrangement
of each world to the **observed** provider probabilities via mean, min and
max. This privileged comparator is an achievable member of that world's
120-tree grammar whenever the true matching is offered. It provides a
reference for attainable search quality. Its probability scores can differ
from the Bayes posterior for the latent target.

## Registered comparisons

Primary contrast is top-two minus random-two IID Brier at 240+4 assessments,
with differences averaged across the two paired worlds in a bank, then a
two-sided paired 95% t interval over 12 independent banks. Require an upper
endpoint below zero for a positive primary result. Report separately top-two
versus top-one (different fit budgets), stock GP and fixed baselines. Report
validation and test losses, training loss of winners, test accuracy, true
grouping in proposal 1 and top 2, and whether the selected tree uses the
true matching. Record candidate assessments, label budgets, tree size,
ranking and construction time, search/validation latency and inference time
on one CPU thread. Count correlated worlds once for uncertainty.

Report IID and shifted results both overall and conditional on whether the
two world labels agree on the shared test row. That stratification occurs
strictly after predictions. Report the observed disagreement fraction and
pair-averaged error floor for any one common probability predictor:
each disagreement alone contributes at least 0.25 to pair-averaged Brier.
The bound concerns a predictor using identical outputs in both worlds.
Neither the helper's label-aware state nor hidden groups cross the boundary
between the two independent fits. Seed-wise output is retained.
