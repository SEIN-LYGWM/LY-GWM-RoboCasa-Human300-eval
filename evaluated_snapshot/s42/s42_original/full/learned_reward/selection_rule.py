"""Experimental predictive-state consensus, not a task reward/value model."""
import math


def select_medoid(states, tie_atol=1e-12):
    """K x 20 NORMALIZED predicted robot states; minimize mean squared distance.

    Each score averages distances to the other K-1 candidates. Equal scores
    prefer the lowest candidate index. No success labels or simulator state.
    """
    if len(states) != 4 or any(len(s) != 20 for s in states):
        raise ValueError('Expected exactly four predicted 20D states')
    if not all(math.isfinite(float(v)) for s in states for v in s):
        raise ValueError('Non-finite future state')
    distances = [[sum((float(a)-float(b))**2 for a,b in zip(x,y))/20
                  for y in states] for x in states]
    scores = [sum(row)/3 for row in distances]
    best = min(scores)
    selected = next(i for i,s in enumerate(scores) if s <= best + tie_atol)
    return selected, scores, max(max(row) for row in distances)


def select_reward(logits):
    """Highest learned reward logit; exact ties choose first. No task oracle."""
    if len(logits)!=4 or not all(math.isfinite(float(x)) for x in logits):
        raise ValueError('Four finite reward logits required')
    return max(range(4), key=lambda i: logits[i])
