# 🧠 Self-Pruning Neural Network

> A neural network that learns to kill its own useless connections during training

## The Story Behind This Code

I came across an interesting challenge: "What if a neural network could figure out which weights matter and which don't—*while* it's training, not after?"

Most pruning happens after training. You train a big model, then go back and chop off the "unimportant" parts. But that always felt backward to me. Why not let the model figure it out in real-time?

So I built this. A 4-layer network where every weight has a little "gatekeeper" (a learnable parameter between 0 and 1). If the gatekeeper decides a weight is useless, it kills it completely. The network literally prunes itself.

## Wait, How Does It Actually Work?

Three simple ideas:

1. **Every weight gets a gate** - For each weight `w`, there's a gate `g` (0 to 1). The real weight used is `w × g`.

2. **L1 penalty pushes gates to zero** - The loss function includes `sum(all gates)`. This constant pressure makes gates want to become 0 (dead).

3. **Classification loss fights back** - If a weight actually helps predict the right answer, it fights to keep its gate alive.

The result? A beautiful battle between "kill everything" (sparsity loss) and "stay useful" (classification loss). The network learns exactly which connections matter.

## What I Learned (That Surprised Me)

**The bimodal distribution blew my mind.** I expected gates to gradually drift toward zero. Instead, they either became 0 (dead) or stayed near 1 (alive). Almost nothing in between. L1 regularization is brutal like that—it doesn't do "kind of dead."

**Mild pruning actually helped accuracy.** At `λ = 0.0001`, the network killed 64% of its weights but *improved* accuracy from 47% to 52%. That's regularization in action—forced simplicity led to better generalization.

**Most weights are useless.** At `λ = 0.001`, accuracy dropped only slightly (47% → 47%) while killing 99.3% of weights. The original network had massive redundancy.

## Results Table

| λ (sparsity strength) | Test Accuracy | Dead Weights |
|----------------------|---------------|--------------|
| 0.0 (baseline) | 47.33% | 0% |
| 0.0001 | 51.96% | 64.2% |
| 0.001 | 46.93% | 99.3% |
| 0.01 | 42.18% | 100% |

## How to Run

```bash
pip install torch torchvision numpy matplotlib tqdm
python self_pruning_network.py
